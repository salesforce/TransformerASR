"""Finetuning methods."""

import logging
import os
import torch

from collections import OrderedDict

from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import torch_load

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.mt_interface import MTInterface

from espnet.utils.dynamic_import import dynamic_import


def transfer_verification(model_state_dict, partial_state_dict, modules):
    """Verify tuples (key, shape) for input model modules match specified modules.

    Args:
        model_state_dict (OrderedDict): the initial model state_dict
        partial_state_dict (OrderedDict): the trained model state_dict
        modules (list): specified module list for transfer

    Return:
        (boolean): allow transfer

    """
    partial_modules = []
    for key_p, value_p in partial_state_dict.items():
        if any(key_p.startswith(m) for m in modules):
            if value_p.shape == model_state_dict[key_p].shape:
                partial_modules += [(key_p, value_p.shape)]
    return len(partial_modules) > 0


def get_partial_asr_mt_state_dict(model_state_dict, modules):
    """Create state_dict with specified modules matching input model modules.

    Args:
        model_state_dict (OrderedDict): trained model state_dict
        modules (list): specified module list for transfer

    Return:
        new_state_dict (OrderedDict): the updated state_dict

    """
    new_state_dict = OrderedDict()

    for key, value in model_state_dict.items():
        if any(key.startswith(m) for m in modules):
            new_state_dict[key] = value

    return new_state_dict


def get_partial_lm_state_dict(model_state_dict, modules):
    """Create compatible ASR state_dict from model_state_dict (LM).

    The keys for specified modules are modified to match ASR decoder modules keys.

    Args:
        model_state_dict (OrderedDict): trained model state_dict
        modules (list): specified module list for transfer

    Return:
        new_state_dict (OrderedDict): the updated state_dict
        new_mods (list): the updated module list

    """
    new_state_dict = OrderedDict()
    new_modules = []

    for key, value in list(model_state_dict.items()):
        if key == "predictor.embed.weight" and "predictor.embed." in modules:
            new_key = "dec.embed.weight"
            new_state_dict[new_key] = value
            new_modules += [new_key]
        elif "predictor.rnn." in key and "predictor.rnn." in modules:
            new_key = "dec.decoder." + key.split("predictor.rnn.", 1)[1]
            new_state_dict[new_key] = value
            new_modules += [new_key]

    return new_state_dict, new_modules


def filter_modules(model_state_dict, modules):
    """Filter non-matched modules in module_state_dict.

    Args:
        model_state_dict (OrderedDict): trained model state_dict
        modules (list): specified module list for transfer

    Return:
        new_mods (list): the update module list

    """
    new_mods = []
    incorrect_mods = []

    mods_model = list(model_state_dict.keys())
    for mod in modules:
        if any(key.startswith(mod) for key in mods_model):
            new_mods += [mod]
        else:
            incorrect_mods += [mod]

    if incorrect_mods:
        logging.warning("module(s) %s don\'t match or (partially match) "
                        "available modules in model.", incorrect_mods)
        logging.warning('for information, the existing modules in model are:')
        logging.warning('%s', mods_model)
        logging.warning("incorrect ones:")
        logging.warning('%s', incorrect_mods)

    return new_mods


def load_trained_model(model_path):
    """Load the trained model for recognition.

    Args:
        model_path (str): Path to model.***.best

    """
    idim, odim, train_args = get_model_conf(
        model_path, os.path.join(os.path.dirname(model_path), 'model.json'))

    logging.warning('reading model parameters from ' + model_path)

    if hasattr(train_args, "model_module"):
        model_module = train_args.model_module
    else:
        model_module = "espnet.nets.pytorch_backend.e2e_asr:E2E"
    model_class = dynamic_import(model_module)
    model = model_class(idim, odim, train_args)

    torch_load(model_path, model)

    return model, train_args


def get_trained_model_state_dict(model_path):
    """Extract the trained model state dict for pre-initialization.

    Args:
        model_path (str): Path to model.***.best

    Return:
        model.state_dict() (OrderedDict): the loaded model state_dict
        (str): Type of model. Either ASR/MT or LM.

    """
    conf_path = os.path.join(os.path.dirname(model_path), 'model.json')
    if 'rnnlm' in model_path:
        logging.warning('reading model parameters from %s', model_path)

        return torch.load(model_path), 'lm'

    idim, odim, args = get_model_conf(model_path, conf_path)

    logging.warning('reading model parameters from ' + model_path)

    if hasattr(args, "model_module"):
        model_module = args.model_module
    else:
        model_module = "espnet.nets.pytorch_backend.e2e_asr:E2E"

    model_class = dynamic_import(model_module)
    model = model_class(idim, odim, args)
    torch_load(model_path, model)
    assert isinstance(model, MTInterface) or isinstance(model, ASRInterface)

    return model.state_dict(), 'asr-mt'


def load_trained_modules(idim, odim, args, interface=ASRInterface):
    """Load model encoder or/and decoder modules with ESPNET pre-trained model(s).

    Args:
        idim (int): initial input dimension.
        odim (int): initial output dimension.
        args (Namespace): The initial model arguments.
        interface (Interface): ASRInterface or STInterface

    Return:
        model (torch.nn.Module): The model with pretrained modules.

    """
    enc_model_path = args.enc_init
    dec_model_path = args.dec_init
    enc_modules = args.enc_init_mods
    dec_modules = args.dec_init_mods
    # Weiran: I added ctc modules to original function.
    if hasattr(args, 'ctc_init'):
        ctc_model_path = args.ctc_init
        ctc_modules = args.ctc_init_mods
    else:
        ctc_model_path = None
        ctc_modules = None

    model_class = dynamic_import(args.model_module)
    main_model = model_class(idim, odim, args)
    assert isinstance(main_model, interface)

    main_state_dict = main_model.state_dict()
    if not enc_modules and hasattr(main_model, 'encoder'):
        enc_modules = list(["encoder." + x for x in main_model.encoder.state_dict().keys()])
        logging.warning("default enc_modules: %s" % enc_modules)
    if not dec_modules and hasattr(main_model, 'decoder'):
        dec_modules = list(["decoder." + x for x in list(main_model.decoder.state_dict().keys())])
        logging.warning("default dec_modules: %s" % dec_modules)
    if not ctc_modules and hasattr(main_model, 'ctc'):
        ctc_modules = list(["ctc." + x for x in list(main_model.ctc.state_dict().keys())])
        logging.warning("default ctc_modules: %s" % ctc_modules)

    logging.warning('model(s) found for pre-initialization')
    for model_path, modules in [(enc_model_path, enc_modules),
                                (dec_model_path, dec_modules),
                                (ctc_model_path, ctc_modules)]:
        if model_path is not None:
            if os.path.isfile(model_path):
                model_state_dict, mode = get_trained_model_state_dict(model_path)
                logging.warning("loading trained model from %s" % model_path)
                # logging.warning("trained model_state_dict: %s" % model_state_dict.keys())

                modules = filter_modules(model_state_dict, modules)
                print("the following modules will be kept: %s" % modules)
                if mode == 'lm':
                    partial_state_dict, modules = get_partial_lm_state_dict(model_state_dict, modules)
                else:
                    partial_state_dict = get_partial_asr_mt_state_dict(model_state_dict, modules)

                    if partial_state_dict:
                        if transfer_verification(main_state_dict, partial_state_dict, modules):
                            logging.warning('loading %s from model: %s', modules, model_path)
                            for k in partial_state_dict.keys():
                                logging.warning('override %s' % k)
                            main_state_dict.update(partial_state_dict)
                        else:
                            logging.warning('modules %s in model %s don\'t match your training config',
                                            modules, model_path)
            else:
                logging.error('model was not found : %s', model_path)
                raise ValueError

    main_model.load_state_dict(main_state_dict)

    return main_model
