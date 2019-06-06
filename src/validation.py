from joblib import load
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools
import networkx as nx
from graphviz import Digraph
import colorsys
from sklearn.metrics import confusion_matrix



def get_inputs_from_selected_measurements(fault_location,selected_feature_ids, all_data, validation_data):


    input_data = validation_data['validation_input']
    output_data = validation_data['validation_output']

    row_indices_of_the_fault = np.where(output_data == fault_location)[0]



    node_feature_ind = [i for i in range(len(all_data['node_ids']))]
    line_feature_ind = [len(all_data['node_ids']) + i for i in range(len(all_data['line_ids']))]

    feature_ids = list(all_data['node_ids'])
    feature_ids.extend(all_data['line_ids'])
    feature_inds = list(node_feature_ind)
    feature_inds.extend(line_feature_ind)

    selected_feature_inds = {}

    for clf_id in selected_feature_ids:
        selected_feature_inds[clf_id] = []
        for id in selected_feature_ids[clf_id]:
            selected_feature_inds[clf_id].append(feature_ids.index(id))

    del all_data

    input_data_sel = {}
    random_row = random.choice(row_indices_of_the_fault)

    for clf_id in selected_feature_inds:
        input_data_sel[clf_id] = []
        for feature in selected_feature_inds[clf_id]:
            input_data_sel[clf_id].append(input_data[random_row, int(feature)])

        #input_data_sel[clf_id] = input_data_sel[clf_id]


    return input_data_sel

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def test_input(fault, dict_of_grids, all_data, validation_data):

    measurement_set = {}
    for grid_id, grid in dict_of_grids.items():
        measurement_set[grid_id] = grid.optimal_feature_ids_for_total_classification[:4]


    randomly_selected_inputs_for_prediction = get_inputs_from_selected_measurements(fault, measurement_set, all_data, validation_data)
    '''
    randomly_selected_inputs_for_prediction = {'r1':[1.5707831926489, 5.75667156305051, 27.9904000024143, 2728.99327161135],
                       'r2':[10.323, 5.75595169882724, 1.5707963267949, 2736.84770438547],
                       'r3':[0.025135201237837, 5.75486387301128, 103487.688919156, 2733.70914341669],
                       'r4':[0,0,0,0],
                       'r5':[0,0,0,0]

    }
    '''
    all_accumulated_inps = []
    for grid_id in dict_of_grids:
        if grid_id != "r":
            all_accumulated_inps.extend(randomly_selected_inputs_for_prediction[grid_id])

    all_accumulated_inps = np.asarray(all_accumulated_inps).reshape(1,-1)

    pred_grid_id = dict_of_grids["r"].zone_classifier_model.predict(all_accumulated_inps)

    print(pred_grid_id[0])

    inp_for_pred_grid = np.asarray(randomly_selected_inputs_for_prediction[pred_grid_id[0]]).reshape(1, -1)

    print(dict_of_grids[pred_grid_id[0]].total_classifier_model.predict(inp_for_pred_grid)[0])

    '''
    for grid_id, grid in dict_of_grids.items():
        clf = grid.total_classifier_model
        inp = np.asarray(randomly_selected_inputs_for_prediction[grid_id]).reshape(1, -1)
        print(clf.predict(inp))
    '''

def plot_acc(grid_id, cnf_matrix, class_names):
    # Compute confusion matrix
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix for: '+grid_id)
    #plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=False,
    #                      title='Non-normalized confusion matrix for: ' + grid_id)

    plt.show()


def get_score(class_labels,cnf_mat,line):

    line_label = str(line[0])+"_"+str(line[1])

    ind = np.where(class_labels==line_label)
    score = float(cnf_mat[ind,ind]/np.sum(cnf_mat[ind,:]))

    print(line_label,score)

    return score

def plot_accuracy_on_grid_gviz(grid_data_dict,all_data):

    g = Digraph()
    g.format = 'svg'

    node_list = []
    line_list = []

    edge_details = {}
    meas_node = []
    meas_line = []
    i = 0

    norm = mpl.colors.Normalize(vmin=0, vmax=len(grid_data_dict))
    cmap = mpl.cm.jet
    m = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)



    node_number = 0
    line_number = 0

    for grid_id, grid in grid_data_dict.items():
        if grid_id != "r":
            feature_id = grid.optimal_feature_ids_for_total_classification[:4]
            for ftr in feature_id:
                if ftr[0] == "n":
                    node_number = int(''.join(list(filter(str.isdigit, ftr))))
                if ftr[0] == "l":
                    line_number = int(''.join(list(filter(str.isdigit, ftr))))


                if node_number not in meas_node:
                    meas_node.append(node_number)
                if all_data['line_dict'][line_number][:2] not in meas_line:
                    meas_line.append(all_data['line_dict'][line_number][:2])


            for node in grid.node_list:
                if node not in node_list:
                    node_list.append(node)
            for line in grid.line_list:
                if line not in line_list:
                    line_list.append(line)
                    line_label = str(line[0]) + "_" + str(line[1])
                    edge_details[line_label] = {}
                    if line_label in grid.confusion_matrix_total:
                        score = grid.confusion_matrix_total[line_label]
                        edge_details[line_label]['width'] = (2*(score)+1)**2
                        edge_details[line_label]['acc'] = round(score,2)
                        mm = m.to_rgba(i)
                        M = colorsys.rgb_to_hsv(mm[0], mm[1], mm[2])
                        edge_details[line_label]['zone_id'] = str(M[0])+" "+str(M[1])+" "+str(M[2])
                    else:
                        edge_details[line_label]['width'] = 1
                        edge_details[line_label]['acc'] = "N.A."
                        mm = m.to_rgba(i)
                        M = colorsys.rgb_to_hsv(mm[0], mm[1], mm[2])
                        edge_details[line_label]['zone_id'] = str(M[0])+" "+str(M[1])+" "+str(M[2])
            i = i + 1


    for node in node_list:
        if node not in meas_node:
            g.node(str(node), shape='circle')
        else:
            g.node(str(node), shape='octagon')



    for line in line_list:
        line_label = str(line[0]) + "_" + str(line[1])
        if line not in meas_line:
            g.edge(str(line[0]), str(line[1]), arrowhead='none', penwidth=str(edge_details[line_label]['width']), label="   "+str(edge_details[line_label]['acc'])+"   ", color = edge_details[line_label]['zone_id'])
        else:
            g.edge(str(line[0]), str(line[1]), arrowhead='none', style='dashed', penwidth=str(edge_details[line_label]['width']),label ="   "+str(edge_details[line_label]['acc'])+"   ", color = edge_details[line_label]['zone_id'])

    g.render('ieee_123_',view=True)


def plot_accuracy_on_grid(grid_data_dict):
    grid_graph = nx.Graph()

    node_list = []
    line_list = []
    node_colour = []
    edge_colour = []
    edge_width = []
    node_shape = []
    meas_node = [708]
    meas_line = [[708, 733]]
    edge_style = []
    i=0
    for grid_id, grid in grid_data_dict.items():
        for node in grid.node_list:
            if node not in node_list:
                node_list.append(node)

                if node not in meas_node:
                    node_colour.append('r')
                    node_shape.append('o')
                else:
                    node_shape.append('d')
                    node_colour.append('b')
        for line in grid.line_list:
            if line not in line_list:
                line_list.append(line)
                line_label = str(line[0]) + "_" + str(line[1])
                score = grid.confusion_matrix[line_label]
                print(line_label,score)
                edge_width.append(10*score)

                if line not in meas_line:
                    edge_colour.append('b')
                    edge_style.append('solid')
                else:
                    edge_style.append('dotted')
                    edge_colour.append('black')
        i=i+1



    for node in node_list:
        grid_graph.add_node(node, label = str(node), node_shape='d')



    for line in line_list:
        grid_graph.add_edge(line[0], line[1], arrowhead='none', width=edge_width)

    pos = nx.nx_agraph.graphviz_layout(grid_graph)



    #nx.draw(grid_graph,pos=pos,node_color=node_colour, edge_color=edge_colours, width=4, edge_cmap=plt.cm.Blues, cmap=plt.cm.Blues, with_labels=False)
    nx.draw(grid_graph,pos=pos, with_labels=True, node_size=1000, node_color=node_colour)


    plt.show()


def main():
    all_data = load(os.getcwd() + "/scenario_data/all_data_123_a.pkl")
    validation_data = load("validation_data.pkl")
    grids = load(os.getcwd() + "/dict_of_grids.pkl")
    fault = "40_42"
    test_input(fault, grids, all_data, validation_data)


    #for grid_id,grid in grids.items():
    #    plot_acc(grid_id,grid.confusion_matrix,grid.total_classifier_model.classes_)

    plot_accuracy_on_grid_gviz(grids,all_data)





if __name__ == "__main__":
    main()


