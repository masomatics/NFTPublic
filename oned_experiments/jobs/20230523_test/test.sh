python ./run.py --config_path=./configs/NDFT.yml --attr batchsize=32 seed=1 max_iteration=100000 report_freq=100 model_snapshot_freq=100000 manager_snapshot_freq=100000 num_workers=2 T_cond=2 T_pred=1 lr=1e-05 reg.reg_bd=None reg.reg_orth=None train_data.fn=./period_fn.py train_data.name=Shifted_FreqFun_nl train_data.args.Ndata=30000 train_data.args.N=128 train_data.args.T=3 train_data.args.max_T=9 train_data.args.shift_label=True train_data.args.shift_range=0.4 train_data.args.batchM_size=20 train_data.args.nfreq=5 train_data.args.freq_fix=True train_data.args.freqseed=1 train_data.args.smallfreqs_num=2 train_data.args.smallfreqs_strength=0.2 model.fn=./models/seqae.py model.name=SeqAETSQmlp model.args.dim_data=128 model.args.dim_m=16 model.args.dim_a=10 model.args.predictive=True model.args.transition_model=LS model.args.activation=tanh training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=80000 training_loop.args.reconst_iter=30000 log_dir=../result/nfa/20230523_test_0/freqseed1


python ./run.py --config_path=./configs/NDFT.yml --attr batchsize=32 seed=1 max_iteration=100000 report_freq=100 model_snapshot_freq=100000 manager_snapshot_freq=100000 num_workers=2 T_cond=2 T_pred=1 lr=1e-05 reg.reg_bd=None reg.reg_orth=None train_data.fn=./period_fn.py train_data.name=Shifted_FreqFun_nl train_data.args.Ndata=30000 train_data.args.N=128 train_data.args.T=3 train_data.args.max_T=9 train_data.args.shift_label=True train_data.args.shift_range=0.4 train_data.args.batchM_size=20 train_data.args.nfreq=5 train_data.args.freq_fix=True train_data.args.freqseed=2 train_data.args.smallfreqs_num=2 train_data.args.smallfreqs_strength=0.2 model.fn=./models/seqae.py model.name=SeqAETSQmlp model.args.dim_data=128 model.args.dim_m=16 model.args.dim_a=10 model.args.predictive=True model.args.transition_model=LS model.args.activation=tanh training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=80000 training_loop.args.reconst_iter=30000 log_dir=../result/nfa/20230523_test_0/freqseed2


python ./run.py --config_path=./configs/NDFT.yml --attr batchsize=32 seed=1 max_iteration=100000 report_freq=100 model_snapshot_freq=100000 manager_snapshot_freq=100000 num_workers=2 T_cond=2 T_pred=1 lr=1e-05 reg.reg_bd=None reg.reg_orth=None train_data.fn=./period_fn.py train_data.name=Shifted_FreqFun_nl train_data.args.Ndata=30000 train_data.args.N=128 train_data.args.T=3 train_data.args.max_T=9 train_data.args.shift_label=True train_data.args.shift_range=0.4 train_data.args.batchM_size=20 train_data.args.nfreq=5 train_data.args.freq_fix=True train_data.args.freqseed=3 train_data.args.smallfreqs_num=2 train_data.args.smallfreqs_strength=0.2 model.fn=./models/seqae.py model.name=SeqAETSQmlp model.args.dim_data=128 model.args.dim_m=16 model.args.dim_a=10 model.args.predictive=True model.args.transition_model=LS model.args.activation=tanh training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=80000 training_loop.args.reconst_iter=30000 log_dir=../result/nfa/20230523_test_0/freqseed3

