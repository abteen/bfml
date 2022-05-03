import argparse, logging, git, os, sys
import numpy, torch
from transformers import set_seed, logging as trf_logging
from omegaconf import OmegaConf
from datetime import datetime


def set_seeds(seed=42):
    set_seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_experiment():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_config', default='configs/empty.yaml')
    parser.add_argument('--experiment_config', default='configs/empty.yaml')
    parser.add_argument('--training_args', default='configs/empty.yaml')
    parser.add_argument('--tokenizer_config', default='configs/empty.yaml')

    args, _ = parser.parse_known_args()

    dataset_cfg = OmegaConf.load(args.dataset_config)
    exp_cfg = OmegaConf.load(args.experiment_config)
    ta_cfg = OmegaConf.load(args.training_args)
    tok_cfg = OmegaConf.load(args.tokenizer_config)
    cli_cfg = OmegaConf.from_cli()

    config = OmegaConf.merge(dataset_cfg, exp_cfg, ta_cfg, tok_cfg, cli_cfg)


    set_seeds(config['seed'])

    intended_log_dir = os.path.join(config['log_directory'], config['experiment_name'], config['group_name'])

    if not os.path.isdir(intended_log_dir):
        os.makedirs(intended_log_dir)

    # Setup logging
    logging.root.handlers = []
    # noinspection PyArgumentList
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("{}/{}.log".format(intended_log_dir, config['run_name'])),
            logging.StreamHandler()
        ]
    )

    trf_logging.set_verbosity_info()
    trf_logging.enable_propagation()

    logging.info('Loaded config: \n{}'.format(OmegaConf.to_yaml(config)))

    #Check if model has been trained already -- if so, exit.
    output_directory = os.path.join(config['output_directory'], config['experiment_name'], config['group_name'], config['run_name'])
    if config.check_already_trained:
        final_model_directory = os.path.join(output_directory, 'final_model')
        if os.path.isdir(final_model_directory) and 'pytorch_model.bin' in os.listdir(final_model_directory):
            logging.info('Final model for this experiment exists, exiting without training')
            sys.exit(0)

    if config.check_git_status:
        repo = git.Repo(search_parent_directories=True)
        commit_id = repo.head.object.hexsha
        branch = repo.active_branch.name
        logging.info('Using code from git commit: {}'.format(commit_id))
        logging.info('On active branch: {}'.format(branch))

    # Set visible devices

    if not config.get('dev', None):
        os.environ['CUDA_VISIBLE_DEVICES'] = config['visible_devices']
        import torch
        try:
            assert torch.cuda.device_count() == config['n_gpu']
        except AssertionError as err:
            logging.error('Expected {} GPUs available, but only see {} (visible devices: {})'.format(config['n_gpu'],
                                                                                                    torch.cuda.device_count(),
                                                                                                    config[
                                                                                                        'visible_devices']))
            sys.exit(1)

        # Setup wandb
        if config['use_wandb']:
            import wandb
            wandb.init(project=config['experiment_name'], group=config['group_name'], name=config['run_name'])
            config['training_arguments']['report_to'] = 'all'
        else:
            os.environ['WANDB_DISABLED'] = 'True'
            config['training_arguments']['report_to'] = 'none'

        time = datetime.now().strftime("%m-%d-%H:%M")
        logging.info('Start time: {}'.format(time))


        logging.info('Number of GPUs available: {}'.format(torch.cuda.device_count()))
        logging.info('Using the following GPUs: {}'.format(config['visible_devices']))

        assert config['expected_batch_size'] == config['n_gpu'] * \
               config['training_arguments']['per_device_train_batch_size'] * \
               config['training_arguments']['gradient_accumulation_steps']

    return args, config, output_directory
