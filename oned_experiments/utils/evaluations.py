import torch
import pdb
from torch.utils.data import DataLoader
from utils import notebook_utils as nu
from utils import yaml_utils as yu
import numpy as np
import os
from tqdm import tqdm



'''
Scripts for running evaluations.

Usage: Set targfile_path to be the folder containing the result. (e.g. ../result/nfa/20230525_projname/expname)

allresults, targ, xnext  = prediction_evaluation([targfile_path], device =0, n_cond=2, 
                                                    tp=tp, repeats=1,predictive=False,
                                                    reconstructive = False,alteration={},
                                                   mode='notebook', testseed=4)

allresults will contain  (1) prediction results (2) regressed matrices (3)models and (4) action-labels that was used in the 
experiment to produce the result.

'''

result_dir = '../result/nfa'


#might be overlapping with get_predict
def predict(images, model,
            n_cond=2, tp=5, device='cpu', swap =False,
            predictive=False, reconstructive=False, label=None):

    if type(images) == list:
        images = torch.stack(images)
        images = images.transpose(1, 0)

    images = images.to(device)
    images_cond = images[:, :n_cond]

    if model.transition_model == 'thetaL':
        M = model.get_M(images_cond,indices=label) 
    else:
        M = model.get_M(images_cond)  # n a a
    if type(M) == tuple:
        M = M[0]

    if predictive:
        H = model.encode(images_cond[:, [0]])[:, 0]
        tp = n_cond -1 + tp
        xs0 = images[:, [0]].to('cpu')
    else:
        H = model.encode(images_cond[:, -1:])[:, 0] # n s a
        xs0 = []
        if reconstructive:
            xs0 = torch.sigmoid(model.decode(model.encode(images_cond[:, :n_cond])).detach().to('cpu'))
    xs = []
    n, s, a = H.shape


    if swap == True:
        M = M[torch.arange(-n//2, n-n//2)]

    for r in range(tp):
        H = H @ M[:H.shape[0]]
        x_next_t = model.decode(H[:, None])
        xs.append(x_next_t)
    #x_next = torch.sigmoid(torch.cat(xs, axis=1).detach().to('cpu'))
    x_next = torch.cat(xs, axis=1).detach().to('cpu')
    if len(xs0) > 0:
        x_next = torch.cat([xs0] +[x_next], axis=1)

    return x_next, M




def prediction_evaluation(targdir_pathlist, device =0,
                           n_cond=2, tp=1, repeats=3,
                           predictive=False,reconstructive = False,
                          alteration={}, 
                          mode='default', testseed=0, examples=False):
    results = {}
    inferred_Ms = {}
    model_configs = {}
    models = {}
    all_configs = {}
    labels = {}

    for targdir_path in targdir_pathlist:

        if os.path.exists(os.path.join(targdir_path, 'config.yml')):
            config = nu.load_config(targdir_path)
        else:
            config = nu.load_config(baseline_path)

        config = yu.alter_config(config, alteration)

        model_config = config['model']
        model_config['args']['dim_data'] = config['train_data']['args']['N']

        model = yu.load_component(model_config)

        iterlist = nu.iter_list(targdir_path)



        if len(iterlist) == 0:
            print(f"""There is no model trained for {targdir_path}""")
        else:
            maxiter = np.max(nu.iter_list(targdir_path))

            try:
                nu.load_model(model, targdir_path, maxiter)
            except:
                pdb.set_trace()
            model = model.eval().to(device)

            with torch.no_grad():
                l2scores = []
                for j in range(repeats):
                    dataconfig = config['train_data']
                    dataconfig['args']['T'] = tp + n_cond
                    dataconfig['args']['max_T'] = tp + n_cond
                    dataconfig['args']['test'] = j + 1 + testseed

                    data = yu.load_component(dataconfig)

                    train_loader = DataLoader(data,
                                            batch_size=config['batchsize'],
                                            shuffle=True,
                                            num_workers=config['num_workers'])    

                    Mlist = []
                    label_list = []
                    index_list = []
                    for images, label in tqdm(train_loader):
                        if type(images) == list:
                            images = torch.stack(images)
                            images = images.transpose(1, 0)
                        # n t c w h
                        images = images.to(device)

                        if predictive == True or reconstructive == True:
                            images_target = images
                        else:
                            images_target = images[:, n_cond:n_cond + tp]
                        x_next, M = predict(images, model, n_cond=n_cond,
                                               tp=tp, device=device,
                                               predictive=predictive,
                                               reconstructive=reconstructive, 
                                               label=label)
                        l2_losses = torch.sum(
                            (images_target.to('cpu') - x_next.to('cpu')) ** 2,
                            axis=[-1])
                        l2scores.append(l2_losses)

                        Mlist.append(M.detach().to('cpu'))
                        if len(label) == 2:
                            label_list.append(label[0])
                            index_list.append(label[1])
                        else:
                            label_list.append(label)

                    Mlist = torch.cat(Mlist)
                    label_list = torch.cat(label_list)
                    if len(index_list) > 0:
                        index_list = torch.cat(index_list)



            l2scores = torch.cat(l2scores)
            av_l2 = torch.mean(l2scores, axis=0)
            av_l2var = torch.std(l2scores, axis=0)
            print(av_l2)
            print(av_l2var)
            results[targdir_path] = [av_l2, av_l2var]

            inferred_Ms[targdir_path] = Mlist
            models[targdir_path] = model.to('cpu')
            model_configs[targdir_path] = model_config
            all_configs[targdir_path] = config
            labels[targdir_path] = label_list

    output={'results':results,
            'Ms': inferred_Ms,
            'configs': all_configs,
            'models': models,
            'labels': labels}

    if examples==True:
        output['examples'] =  {'targ':  images_target.to('cpu'), 
        'est': x_next.to('cpu'), 'label':label}    

    return output, images_target.to('cpu'), x_next.to('cpu')


def get_predict(images, targdir_path, swap=False, predictive=False,device=0,
                n_cond=2, tp=1):
    if os.path.exists(os.path.join(targdir_path, 'config.yml')):
        config = nu.load_config(targdir_path)
    else:
        config = nu.load_config(baseline_path)

    # config = load_config(targdir_path)

    model_config = config['model']
    if len(nu.iter_list(targdir_path)) > 0:
        maxiter = np.max(nu.iter_list(targdir_path))
        model = yu.load_component(model_config).to(device)
        nu.load_model(model, targdir_path, maxiter)
        model = model.eval().to(device)
        # model(images[:, :2])
        if str(type(model)).split(' ')[-1].split('.')[-1].split("'")[
            0] == 'SeqAENeuralM_latentPredict':
            model.conduct_prediction(images[:, :n_cond], n_rolls=tp)
        else:
            model(images[:, :n_cond])
        x_next, M = predict(images, model, n_cond=n_cond, tp=tp,
                               device=device, swap=swap,
                               predictive=predictive)
        return x_next, M
    else:
        return 0, 0

