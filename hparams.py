
class Seq_Len:
    seq_len_1 = 2
    seq_1 = 'window_1'
    seq_len_4 = 5
    seq_4 = 'window_4'
    seq_len_8 = 9
    seq_8 = 'window_8'
    seq_len_16 = 17
    seq_16 = 'window_16'

class NonPrePaddedRobotFrame_Datasets_Timestep_0_5:
    data_dir = 'RobotFrameDatasetsTimestep05'
    time_steps =  400
    RNN_runs = 'mainNonPrePaddedRobotFrameDatasetsTimestep05'
    ckpt_dir = 'ckpt'
    rnnsave = 'mainNonPrePaddedRobotFrameDatasetsTimestep05'

class NonPrePaddedRobotFrame_Datasets_Timestep_0_25:
    data_dir = 'RobotFrameDatasetsTimestep025'
    time_steps =  800
    RNN_runs = 'mainNonPrePaddedRobotFrameDatasetsTimestep025'
    ckpt_dir = 'ckpt'
    rnnsave = 'mainNonPrePaddedRobotFrameDatasetsTimestep025'

class NonPrePaddedWorldFrame_Datasets_Timestep_0_5:
    data_dir = 'WorldFrameDatasetsTimestep05'
    time_steps =  400
    RNN_runs = 'NonPrePaddedWorldFrameDatasetsTimestep05'
    ckpt_dir = 'ckpt'
    rnnsave = 'NonPrePaddedWorldFrameDatasetsTimestep05'

class NonPrePaddedWorldFrame_Datasets_Timestep_0_25:
    data_dir = 'WorldFrameDatasetsTimestep025'
    time_steps =  800
    RNN_runs = 'NonPrePaddedWorldFrameDatasetsTimestep025'
    ckpt_dir = 'ckpt'
    rnnsave = 'NonPrePaddedWorldFrameDatasetsTimestep025'

class NonPrePaddedRobotFrame_Datasets_Timestep_1:
    data_dir = 'RobotFrameDatasetsTimestep1'
    time_steps =  200
    RNN_runs = 'mainNonPrePaddedRobotFrameDatasetsTimestep1'
    ckpt_dir = 'ckpt'
    rnnsave = 'mainNonPrePaddedRobotFrameDatasetsTimestep1'

class NonPrePaddedRobotFrame_Datasets_Timestep_2:
    data_dir = 'RobotFrameDatasetsTimestep2'
    time_steps =  100
    RNN_runs = 'mainNonPrePaddedRobotFrameDatasetsTimestep2'
    ckpt_dir = 'ckpt'
    rnnsave = 'mainNonPrePaddedRobotFrameDatasetsTimestep2'



class WorldFrame_Datasets_Timestep_2:
    data_dir = 'WorldFrameDatasetsTimestep2'
    time_steps =  100
    RNN_runs = 'WorldFrameDatasetsTimestep2'
    ckpt_dir = 'ckpt'
    rnnsave = 'WorldFrameDatasetsTimestep2'


class WorldFrame_Datasets_Timestep_1:
    data_dir = 'WorldFrameDatasetsTimestep1'
    time_steps =  200
    RNN_runs = 'WorldFrameDatasetsTimestep1'
    ckpt_dir = 'ckpt'
    rnnsave = 'WorldFrameDatasetsTimestep1'

class WorldFrame_Datasets_Timestep_0_5:
    data_dir = 'WorldFrameDatasetsTimestep05'
    time_steps =  400
    RNN_runs = 'WorldFrameDatasetsTimestep05'
    ckpt_dir = 'ckpt'
    rnnsave = 'WorldFrameDatasetsTimestep05'

class WorldFrame_Datasets_Timestep_0_25:
    data_dir = 'WorldFrameDatasetsTimestep025'
    time_steps =  800
    RNN_runs = 'WorldFrameDatasetsTimestep025'
    ckpt_dir = 'ckpt'
    rnnsave = 'WorldFrameDatasetsTimestep025'


class RobotFrame_Datasets_Timestep_2:
    data_dir = 'RobotFrameDatasetsTimestep2'
    time_steps =  100
    RNN_runs = 'RobotFrameDatasetsTimestep2'
    ckpt_dir = 'ckpt'
    rnnsave = 'RobotFrameDatasetsTimestep2'

class RobotFrame_Datasets_Timestep_1:
    data_dir = 'RobotFrameDatasetsTimestep1'
    time_steps =  200
    RNN_runs = 'RobotFrameDatasetsTimestep1'
    ckpt_dir = 'ckpt'
    rnnsave = 'RobotFrameDatasetsTimestep1'

class RobotFrame_Datasets_Timestep_0_5:
    data_dir = 'RobotFrameDatasetsTimestep05'
    time_steps =  400
    RNN_runs = 'RobotFrameDatasetsTimestep05'
    ckpt_dir = 'ckpt'
    rnnsave = 'RobotFrameDatasetsTimestep05'

class RobotFrame_Datasets_Timestep_0_25:
    data_dir = 'RobotFrameDatasetsTimestep025'
    time_steps =  800
    RNN_runs = 'RobotFrameDatasetsTimestep025'
    ckpt_dir = 'ckpt'
    rnnsave = 'RobotFrameDatasetsTimestep025'

class DQN_RobotFrame_Datasets_Timestep_1:
    data_dir = 'DQN_RobotFrameDatasetsTimestep1'
    time_steps =  200
    RNN_runs = 'DQN_RobotFrameDatasetsTimestep1'
    ckpt_dir = 'ckpt'
    rnnsave = 'DQN_RobotFrameDatasetsTimestep1'

class DQN_RobotFrame_Datasets_Timestep_0_5:
    data_dir = 'DQN_RobotFrameDatasetsTimestep05'
    time_steps =  400
    RNN_runs = 'DQN_RobotFrameDatasetsTimestep05'
    ckpt_dir = 'ckpt'
    rnnsave = 'DQN_RobotFrameDatasetsTimestep05'


class HyperParams:
    vision = 'VAE'
    memory = 'RNN'
    controller = 'A3C'

    extra = False
    data_dir = 'Datasets'
    extra_dir = 'additional'
    ckpt_dir = 'ckpt'


    batch_size = 2 # actually batchsize * Seqlen
    seq_len = 10

    test_batch = 1
    n_sample = 64

    vsize = 47 # latent size of Vision
    msize = 128 # size of Memory
    asize = 2 # action size
    rnn_hunits = 256
    ctrl_hidden_dims = 512
    log_interval = 5000
    save_interval = 50

    use_binary_feature = False
    score_cut = 300 # to save
    save_start_score = 100

    # Rollout
    max_ep = 300
    n_rollout = 5000
    seed = 0

    n_workers = 0




class WorldFrameHyperParams:
    vision = 'VAE'
    memory = 'RNN'
    controller = 'A3C'

    extra = False
    # data_dir = 'Datasetsworldframe'
    extra_dir = 'additional'
    ckpt_dir = 'ckpt'


    batch_size = 2 # actually batchsize * Seqlen
    # seq_len = 10

    test_batch = 1
    n_sample = 64

    vsize = 53 # latent size of Vision
    msize = 128 # size of Memory
    asize = 2 # action size
    rnn_hunits = 256
    ctrl_hidden_dims = 512
    log_interval = 5000
    save_interval = 50

    use_binary_feature = False
    score_cut = 300 # to save
    save_start_score = 100

    # Rollout
    max_ep = 300
    n_rollout = 5000
    seed = 0

    n_workers = 0

class DQNHyperParams:
    vision = 'VAE'
    memory = 'RNN'
    controller = 'A3C'

    extra = False
    # data_dir = 'dqnDatasets'
    extra_dir = 'additional'
    ckpt_dir = 'ckpt'


    batch_size = 2 # actually batchsize * Seqlen
    seq_len = 10

    test_batch = 1
    n_sample = 64

    vsize = 47 # latent size of Vision
    msize = 128 # size of Memory
    asize = 2 # action size
    rnn_hunits = 256
    ctrl_hidden_dims = 512
    log_interval = 5000
    save_interval = 50

    use_binary_feature = False
    score_cut = 300 # to save
    save_start_score = 100

    # Rollout
    max_ep = 300
    n_rollout = 5000
    seed = 0

    n_workers = 0

class RNNHyperParams:
    vision = 'VAE'
    memory = 'RNN'
    n_hiddens = 256
    extra = False
    # data_dir = 'Datasets'
    extra_dir = 'additional'
    ckpt_dir = 'ckpt'
    seed = 0

    batch_size = 64 # actually batchsize * Seqlen
    test_batch = 1
    seq_len = 10
    n_sample = 47

    vsize = 47#128 # latent size of Vision
    msize = 128 # size of Memory
    asize = 2 # action size
    rnn_hunits = 256
    log_interval = 100
    save_interval = 50

    max_step = 100000

    n_workers = 0


class DQNRNNHyperParams:
    vision = 'VAE'
    memory = 'RNN'
    n_hiddens = 256
    extra = False
    # data_dir = 'dqnDatasets'
    extra_dir = 'additional'
    ckpt_dir = 'ckpt'
    seed = 0    

    

    batch_size = 64 # actually batchsize * Seqlen
    test_batch = 1
    seq_len = 10
    n_sample = 47

    vsize = 47#128 # latent size of Vision
    msize = 128 # size of Memory
    asize = 2 # action size
    rnn_hunits = 256
    log_interval = 100
    save_interval = 50

    max_step = 100000

    n_workers = 0



class VAEHyperParams:
    vision = 'VAE'

    extra = False
    data_dir = 'Datasets'
    extra_dir = 'additional'
    ckpt_dir = 'ckpt'

    n_hiddens = 256
    batch_size = 64 # 
    test_batch = 12
    n_sample = 64

    vsize = 47 # latent size of Vision
    msize = 128 # size of Memory
    asize = 2 # action size

    log_interval = 5000
    save_interval = 10000

    max_step = 100_0000

    n_workers = 0
