import random
from collections import defaultdict

from torch.autograd import Variable
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from tensorboardX import SummaryWriter
from annoy import AnnoyIndex
import networkx as nx

from datasets.ds_movielens import *
from datasets.ds_bookcrossing import *
from models.feedforward.feedforward import FeedforwardModel
from models.heteedge.heteedge_model import *
from models.heteedge.heteedge_aggregation import *
from models.heteedge.heteedge_encoder import *
from models.hetegraphrec.hetegraphrec_aggregation import *
from models.hetegraphrec.hetegraphrec_model import *
from models.hetegraphrec.hetegraphrec_sampler import *
from models.hetegraphrec.hetegraphrec_loss import *
from utils.libs.pytorchtools import *

from cikm import run
from cikm import evaluate_pred
from surprise import Prediction

"""
HeteGraphRec models for different recommendation tasks
"""

# torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)

########################################
########################################


global_step = 1
movielens_processed_data_name = 'hetegraphrec_movielens_ds.data'
bookcrossing_processed_data_name = 'hetegraphrec_bookcrossing_ds.data'

saved_objs_path = 'tmp/saved_objs'
checkpoint_path = 'tmp/checkpoints'
tensorboard_path = 'tmp/tensorboard_runs'
dataset_root_path = '~/data'


def increase_global_step():
    global global_step
    global_step += 1


def save_model(model, path):
    torch.save(model, path)


def load_checkpoint(path):
    return torch.load(path)


def get_dataset(ds_name='movielens'):
    if ds_name == 'movielens':
        dataset_path = dataset_root_path + '/ml-100k'
        processed_data_name = saved_objs_path + '/' + movielens_processed_data_name
        ds = utils.load_obj(processed_data_name)
        if ds is None:
            ds = MovielensDataset(dataset_path)
            ds.process_ratings()
            utils.save_obj(ds, processed_data_name)
    if ds_name == 'bookcrossing':
        dataset_path = dataset_root_path + '/BX-CSV-Dump'
        processed_data_name = saved_objs_path + '/' + bookcrossing_processed_data_name
        ds = utils.load_obj(processed_data_name)
        if ds is None:
            ds = BookCrossingDataset(dataset_path)
            ds.process_ratings()
            utils.save_obj(ds, processed_data_name)
    return ds


def get_tensorboard_logger(comment):
    logger = SummaryWriter(f'{tensorboard_path}/hetegraphrec_{comment}')
    return logger


def get_params_obj(model, params):
    params_obj = {}
    lr = params.get('lr', 0.001)
    params_obj['lr'] = lr
    params_obj['batch_size'] = params.get('batch_size', 512)
    params_obj['epochs'] = params.get('epochs', 10)
    params_obj['train_style'] = params.get('train_style', 'random')
    params_obj['criterion'] = params.get('criterion', 'mse')

    optim = params.get('optimizer', 'sgd')
    optimizer = None
    if optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optim == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    elif optim == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
    elif optim == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    params_obj['optimizer'] = optimizer

    return params_obj


def get_model_heteGraphRec(ds, embedding='edge',
                           pooling='max',
                           two_layer=False,
                           sampler_path={'user': ['movie', 'user'], 'movie': ['user', 'movie']},
                           edge_scores=[1, 2, 3, 4, 5]):
    sampler1 = HeteGraphRecConcatNodeSampler(ds,
                                             path=sampler_path,
                                             edge_weight_choice='highest',
                                             edge_scores=edge_scores)
    input_dim = ds.shape[1]
    pool_hidden_dim1 = 512
    pool_hidden_dim2 = 256
    output_dim1 = 128
    output_dim2 = 64
    if pooling == 'max':
        agg1 = HeteGraphRecNodeMaxPoolAggregator(input_dim,
                                                 output_dim1,
                                                 sampler1,
                                                 combine_type='concat',
                                                 pool_input_dim=input_dim * 2,
                                                 pool_hidden_dim=pool_hidden_dim1)
        # if use combine_type 'concat', final output of agg1 is ouput_dim1 * 2
        if two_layer:
            agg2 = HeteGraphRecNodeMaxPoolAggregator(output_dim1 * 2,
                                                     output_dim2,
                                                     sampler1,
                                                     pre_aggregator=agg1,
                                                     combine_type='concat',
                                                     pool_input_dim=output_dim1 * 4,
                                                     pool_hidden_dim=pool_hidden_dim2)
            # if use combine_type 'concat', final output of agg2 is ouput_dim2 * 2

    if pooling == 'mean':
        agg1 = HeteGraphRecNodeMeanPoolAggregator(input_dim,
                                                  output_dim1,
                                                  sampler1,
                                                  combine_type='concat',
                                                  pool_input_dim=input_dim * 2,
                                                  pool_hidden_dim=pool_hidden_dim1)
        if two_layer:
            agg2 = HeteGraphRecNodeMeanPoolAggregator(output_dim1 * 2,
                                                      output_dim2,
                                                      sampler1,
                                                      pre_aggregator=agg1,
                                                      combine_type='concat',
                                                      pool_input_dim=output_dim1 * 4,
                                                      pool_hidden_dim=pool_hidden_dim2)
    agg = agg1
    output_dim = output_dim1
    if two_layer:
        agg = agg2
        output_dim = output_dim2
    if embedding == 'edge':
        # if use 'concat' combine type, next output layer size will get  double
        model = HeteGraphRecEdgeRegression(output_dim * 2, 1, agg, sampler1, combine_type='concat')
    if embedding == 'node':
        model = HeteGraphRecNodeEmbedding(output_dim * 2, 16, 8, agg, sampler1)
    return model


def get_model_heteEdge(ds):
    dropout_value = 0.5
    agg1 = HeteEdgeMeanAggregator(ds, layer=1)
    enc1 = HeteEdgeEncoder(43, 2048, agg1, dropout_value=dropout_value, layer=1)
    agg2 = HeteEdgeMeanAggregator(ds, enc1, layer=2)
    enc2 = HeteEdgeEncoder(enc1.embed_dim, 1024, agg2, dropout_value=dropout_value, layer=2)
    edge_net = MovielensSuperviseGraphSageModel(1, enc2)

    return edge_net


def get_model_feedforward(ds):
    model_config = {'layers': [('linear', ds.shape[1], 64),
                               ('relu'),
                               ('linear', 64, 32),
                               ('relu'),
                               ('linear', 32, 1)]}
    forward_net = FeedforwardModel(model_config)

    return forward_net


def train_fit_heteEdge_model(ds, model, tensorboard_logger, **params):
    epochs = params['epochs']
    optimizer = params['optimizer']
    criterion = params['criterion']
    batch_size = params['batch_size']

    # train
    ml = get_dataset()
    X_edges, y = ml.get_Xedges_y()
    X_train, X_test, y_train, y_test = train_test_split(X_edges, y, test_size=0.33, random_state=42)

    total_train_sample = len(X_train)
    for epoch in range(epochs):
        random_indices = np.random.choice(total_train_sample, batch_size)
        labels = []
        train_edges = []
        for index in random_indices:
            train_edges.append(X_train[index])
            labels.append(y_train[index])

        optimizer.zero_grad()
        outputs = model(train_edges)
        loss = criterion(outputs, torch.Tensor(labels))
        loss.backward(retain_graph=True)
        optimizer.step()
        print(f'epoch {epoch}, loss {loss}')

    # test
    with torch.no_grad():
        random_indices = np.random.choice(len(X_test), 1024)
        test_labels = []
        test_edges = []
        for index in random_indices:
            test_edges.append(X_test[index])
            test_labels.append(y_test[index])

        predictions = model(test_edges)
        predictions_round = np.rint(predictions.numpy())
        tensor_result = predictions - torch.Tensor(test_labels)
        tensor_result_round = torch.Tensor(predictions_round) - torch.Tensor(test_labels)
        rmse = torch.sqrt(torch.mean(torch.mul(tensor_result, tensor_result)))
        rmse_round = torch.sqrt(torch.mean(torch.mul(tensor_result_round, tensor_result_round)))
        print(f'rmse:{rmse}, rmse_round{rmse_round}')
        utils.hist_plot(test_labels, predictions.numpy(), predictions_round)


def train_fit_feedforward_model(ds, model, tensorboard_logger, **params):
    epochs = params['epochs']
    optimizer = params['optimizer']
    criterion = params['criterion']

    X, y = ds.get_X_y()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    total_sample = X_train.shape[0]
    for epoch in range(epochs):
        features = Variable(torch.Tensor(X_train[epoch:int(total_sample / epochs * (epoch + 1)), :]))
        labels = Variable(torch.Tensor(y_train[epoch:int(total_sample / epochs * (epoch + 1)), :])).squeeze()

        optimizer.zero_grad()
        outputs = model(features)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f'epoch {epoch}, loss {loss}')

    # test
    with torch.no_grad():
        predictions = model(torch.Tensor(X_test))
        preds_round = np.rint(predictions.numpy())
        tensor_result = predictions - torch.Tensor(y_test)
        tensor_result_round = torch.Tensor(preds_round) - torch.Tensor(y_test)
        rmse = torch.sqrt(torch.mean(torch.mul(tensor_result, tensor_result)))
        rmse_round = torch.sqrt(torch.mean(torch.mul(tensor_result_round, tensor_result_round)))
        print(f'rmse:{rmse}, rmse_round{rmse_round}')
        utils.hist_plot(y_test, predictions.numpy(), preds_round)
        print(f'f1 score: {f1_score(y_test, predictions)}')


def train_fit_test_heteGraphRec_model(ds, model, tensorboard_logger, save_name, **params):
    epochs = params['epochs']
    optimizer = params['optimizer']
    criterion = params['criterion']
    train_style = params['train_style']
    batch_size = params['batch_size']

    if criterion == 'mse':
        loss_func = nn.MSELoss()

    # train
    X_edges, y = ds.get_Xedges_y()
    X_train_edges, X_test_edges, y_train, y_test = train_test_split(X_edges, y, test_size=0.33, random_state=42)
    total_train_sample = len(X_train_edges)

    early_stopping = EarlyStopping(patience=15, verbose=True, save_path=checkpoint_path + '/' + save_name)
    for epoch in range(epochs):
        if train_style == 'random':
            random_indices = np.random.choice(total_train_sample, batch_size)
            labels = []
            train_edges = []
            for index in random_indices:
                train_edges.append(X_train_edges[index])
                labels.append(y_train[index])

        elif train_style == 'sequence':
            current_index = (epoch + 1) * batch_size
            if current_index > total_train_sample:
                pre_index = epoch * batch_size
                last_index = total_train_sample - 1
                train_edges = X_train_edges[pre_index:last_index]
                labels = y_train[pre_index:last_index]
            else:
                train_edges = X_train_edges[epoch * batch_size: (epoch + 1) * batch_size]
                labels = y_train[epoch * batch_size: (epoch + 1) * batch_size]

        optimizer.zero_grad()

        model = model.train()
        if criterion == 'mse':
            unique_nodes_embedding, unique_nodes = model(train_edges)
            loss = loss_func(unique_nodes_embedding, torch.Tensor(labels))
        elif criterion == 'max_margin':
            if isinstance(ds, MovielensDataset):
                negative_types = {'user': 'movie', 'movie': 'user'}
            if isinstance(ds, BookCrossingDataset):
                negative_types = {'user': 'book', 'book': 'user'}
            model(train_edges, negative_types)
            loss = model.loss(margin=0.1)

        tensorboard_logger.add_scalar('train_loss', loss, epoch)
        loss.backward()
        optimizer.step()
        print(f'epoch {epoch}, loss {loss}')
        early_stopping(loss.item(), model, save_checkpoint=True)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    model = model.eval()
    if criterion == 'mse':
        try:
            with torch.no_grad():
                rating_prediction, nodes = model(X_test_edges)
                y_test_tensor = torch.Tensor(y_test)
                test_loss = loss_func(rating_prediction, y_test_tensor)
                tensorboard_logger.add_scalar('test_loss', test_loss, global_step)

                predictions_round = np.rint(rating_prediction.numpy())
                tensor_result = rating_prediction - y_test_tensor
                tensor_result_round = torch.Tensor(predictions_round) - y_test_tensor
                rmse = torch.sqrt(torch.mean(torch.mul(tensor_result, tensor_result)))
                rmse_round = torch.sqrt(torch.mean(torch.mul(tensor_result_round, tensor_result_round)))
                print(f'rmse:{rmse}, rmse_round{rmse_round}')

                tensorboard_logger.add_scalar('rmse', rmse, global_step)
                tensorboard_logger.add_scalar('rmse_round', rmse_round, global_step)
                tensorboard_logger.add_histogram('hist', rating_prediction.numpy(), 8)
        except Exception as e:
            print(e)
            return

    elif criterion == 'max_margin':
        print('testing...')
        # with torch.no_grad():
        #     test_embeddings, test_nodes = model(X_test_edges[:256])
        #     a = 0


def run_hetegraphrec_gridsearch(embedding='edge', ds_name='movielens', pooling='max', two_layer=False):
    if embedding == 'edge':
        criterion = ['mse']
    if embedding == 'node':
        criterion = ['max_margin']

    if ds_name == 'movielens':
        ds = get_dataset()
        model = get_model_heteGraphRec(ds,
                                       two_layer=two_layer,
                                       embedding=embedding,
                                       pooling=pooling)

    if ds_name == 'bookcrossing':
        ds = get_dataset(ds_name='bookcrossing')
        model = get_model_heteGraphRec(ds,
                                       two_layer=two_layer,
                                       embedding=embedding,
                                       pooling=pooling,
                                       sampler_path={'user': ['book', 'user'], 'book': ['user', 'book']},
                                       edge_scores=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    grids = {
        'criterion': criterion,
        'lr': [0.001, 0.0005],
        'epochs': [150],
        'optimizer': ['adam', 'sgd', 'rmsprop', 'adadelta', 'adagrad'],
        'batch_size': [512],
        'train_style': ['random']
    }
    for lr in grids['lr']:
        for epochs in grids['epochs']:
            for optim in grids['optimizer']:
                crit = grids['criterion'][0]
                params = {
                    'criterion': crit,
                    'lr': lr,
                    'epochs': epochs,
                    'optimizer': optim,
                    'batch_size': grids['batch_size'][0],
                    'train_style': grids['train_style'][0]
                }
                layer = 1
                if two_layer:
                    layer = 2
                params_obj = get_params_obj(model, params)
                save_name = f'{ds_name}_{pooling}_{embedding}_{layer}_{crit}_{optim}_{lr}'
                tensorboard_logger = get_tensorboard_logger(save_name)
                train_fit_test_heteGraphRec_model(ds, model, tensorboard_logger, save_name, **params_obj)
                increase_global_step()


def test_save_hetegraphrec_embedding(save_name, checkpoint_name, dimension=8, build_tree_size=20):
    ds = get_dataset()
    model = load_checkpoint(checkpoint_path + '/' + checkpoint_name)
    model.eval()

    X_edges, y = ds.get_Xedges_y()
    negative_types = {'user': 'movie', 'movie': 'user'}

    embeddings, nodes = model(random.sample(X_edges, 2048), negative_types)
    t = AnnoyIndex(dimension, metric='euclidean')
    for idx, node in enumerate(nodes):
        t.add_item(node, embeddings[idx])

    t.build(build_tree_size)
    t.save(saved_objs_path + '/' + save_name)


def cross_community_search(save_name, dimension=8):
    predictions = []
    t = AnnoyIndex(dimension, metric='euclidean')
    t.load(saved_objs_path + '/' + save_name)

    user_threshold = 1682
    close_nb_amount = 30
    actual_nb_amount = 20
    ds = get_dataset()
    g = ds.original_graph
    for src_node in range(1682, 1682 + 942):
        nn_nodes, distances = t.get_nns_by_item(src_node, close_nb_amount, include_distances=True)
        each_user_close_nb = {}
        for idx, dst_node in enumerate(nn_nodes):
            # remove non-user node
            if dst_node - user_threshold > 0:
                continue

            # cal weigh of each path
            paths = nx.shortest_path(g, source=src_node, target=dst_node, weight='score')
            path_length = nx.shortest_path_length(g, source=src_node, target=dst_node, weight='score')
            if path_length == 0:
                continue
            print(idx, paths, path_length)
            real_rating = g[src_node][paths[1]]['score']
            pred_rating = path_length / len(paths)
            pred = Prediction(src_node, paths[1], real_rating, pred_rating, {})
            predictions.append(pred)

    rmse, prec, rec, ils_sim = evaluate_pred(g, predictions)
    ds_name = 'movielens'
    algo_name='GraphRec'
    with open(f'eval_{ds_name}.csv', 'a') as f:
        f.write(f'{ds_name}_{algo_name},rmse,{rmse},precision,{prec},recall,{rec},ils,{ils_sim}\n')


def measure_precision_recall(ds_name, k=5, k_score=3, iterations=10):
    ds = get_dataset(ds_name)
    X_edges, y = ds.get_Xedges_y()
    X_train_edges, X_test_edges, y_train, y_test = train_test_split(X_edges, y, test_size=0.33, random_state=42)
    Xtests = sorted(X_test_edges, key=lambda x: x[0])

    model = load_checkpoint(f'{checkpoint_path}/movielens_mean_edge_mse_adam_0.0005_1.133652')
    model.eval()
    with torch.no_grad():
        score_dict = defaultdict(dict)
        for i in range(iterations):
            user_node = random.randint(1682, 1682 + 982)
            edges = filter(lambda x: x[1] == user_node, Xtests)
            small_test_edges = []
            for test_edge in edges:
                small_test_edges.append(test_edge)
            nb_dict = ds.graph[user_node]
            preds, pred_edges = model(small_test_edges)
            for idx, (mv, user) in enumerate(small_test_edges):
                pred = preds[pred_edges.index((mv, user))].item()
                real = nb_dict[mv]['score']
                score_dict[user][mv] = [real, pred]

        for user, user_score in score_dict.items():
            user_score_sort = sorted(user_score.values(), key=lambda x: x[0])
            if len(user_score) >= k:
                pass
        a = 0


if __name__ == "__main__":
    # run_hetegraphrec_gridsearch(embedding='edge', ds_name='movielens', pooling='mean', two_layer=True)
    # run_hetegraphrec_gridsearch(embedding='edge', ds_name='movielens', pooling='max', two_layer=True)
    # run_hetegraphrec_gridsearch(embedding='edge', ds_name='bookcrossing', pooling='max', two_layer=True)
    # run_hetegraphrec_gridsearch(embedding='edge', ds_name='bookcrossing', pooling='mean', two_layer=True)
    #
    # run_hetegraphrec_gridsearch(embedding='edge', ds_name='movielens', pooling='mean', two_layer=False)
    # run_hetegraphrec_gridsearch(embedding='edge', ds_name='movielens', pooling='max', two_layer=False)
    # run_hetegraphrec_gridsearch(embedding='edge', ds_name='bookcrossing', pooling='max', two_layer=False)
    # run_hetegraphrec_gridsearch(embedding='edge', ds_name='bookcrossing', pooling='mean', two_layer=False)
    #
    # run_hetegraphrec_gridsearch(embedding='node', ds_name='movielens', pooling='mean', two_layer=False)
    # run_hetegraphrec_gridsearch(embedding='node', ds_name='movielens', pooling='max', two_layer=False)
    # run_hetegraphrec_gridsearch(embedding='node', ds_name='bookcrossing', pooling='max', two_layer=False)
    # run_hetegraphrec_gridsearch(embedding='node', ds_name='bookcrossing', pooling='mean', two_layer=False)
    #
    # run_hetegraphrec_gridsearch(embedding='node', ds_name='movielens', pooling='mean', two_layer=True)
    # run_hetegraphrec_gridsearch(embedding='node', ds_name='movielens', pooling='max', two_layer=True)
    # run_hetegraphrec_gridsearch(embedding='node', ds_name='bookcrossing', pooling='max', two_layer=True)
    # run_hetegraphrec_gridsearch(embedding='node', ds_name='bookcrossing', pooling='mean', two_layer=True)

    # measure_precision_recall('movielens')

    # run(get_dataset(ds_name='movielens'), 'movielens')

    ann_name = 'test.ann'
    cp = 'movielens_mean_node_1_max_margin_adam_0.0005_20.9722118378'
    emb_dimension = 8
    build_tree_size = 50
    # test_save_hetegraphrec_embedding(ann_name, cp, emb_dimension, build_tree_size)
    cross_community_search(ann_name, emb_dimension)

    run(get_dataset(ds_name='movielens'), 'movielens')
    run(get_dataset(ds_name='bookcrossing'), 'bookcrossing')
