import networkx as nx

class Grid():
    ######### this class stores all the information about any grid that are vital to the operation of FLISR
    def __init__(self,parent_grid_obj=None):



        self.status = "Creating grid"
        print(self.status)
        #### the following are populated during the pipeline execution and just after the grid data ('all_data.pkl') is loaded from the disk
        self.grid_id = ""
        self.node_list = []
        self.line_dict = {}
        self.node_feature_ids = []
        self.line_feature_ids = []
        self.feature_ids = [] ### all feature ids of the grid
        self.feature_inds = [] ### all feature indices of the grid
        self.grid_type = ""  ## should come from a pre-defined list of strings e.g. ['radial','ring','meshed']
        self.node_dict = {}  ## a dictionary containing nodes as keys and list of line dicts
        self.line_list = []  ## a list of lists, where each list consists of a pair of nodes being connected by that line
        self.optimal_locations_for_split = None ## [{node1:[line1,fb1,tb1], node2:[line2,fb2,tb2],..}]
        self.optimal_locations_for_total_classification = []  ## could be a list of dicts (with keys representing nodes), where each dict can have a list (of lines)
        self.grid_graph = None
        self.total_clf_labels_dict = {} ## this dict helps to transform the output of the parent grid to the output labels of the child grids

        #### the following are populated after feature selection
        self.optimal_feature_inds_for_total_classification = [] ### assumed to be of length 4 but can be more, it is constrained-sorted list of feature_inds
        self.optimal_feature_ids_for_total_classification = []

        #### the following are populated after training
        self.total_classification_accuracy = 0
        self.confusion_matrix_total = None
        self.confusion_matrix_zone = None
        self.total_classifier_model = None
        self.total_classifier_model_training_time = 0
        self.zone_classifier_model = None
        self.zone_classifier_model_training_time = 0
        self.zone_classification_accuracy = 0


        ##### the following attributes are internally managed ##########
        #self.__parent_grid_obj__ = parent_grid_obj  ### gives a reference to parent grid object
        if parent_grid_obj== None:
            #self.__parent_grid_obj__ = self  ### gives a reference to parent grid object
            self.parent_grid_id = ""
        else:
            self.parent_grid_id = parent_grid_obj.grid_id

        self.status = "Grid created"
        print(self.status)


    def see_values(self,save_to_file=False):

        details = []

        details.append("Grid ID: "+self.grid_id)
        if self.grid_id == "r":
            details.append("Zone clfn acc: " + str(round(self.zone_classification_accuracy*100,3)) + "%")
        details.append("Total clfn acc: " + str(round(self.total_classification_accuracy * 100, 3)) + "%")
        details.append(
            "Optimal feature ids for total clfn: " + str(self.optimal_feature_ids_for_total_classification[:4]))
        details.append("Parent Grid ID: " + self.parent_grid_id)
        details.append("Optimal locations for split: "+str(self.optimal_locations_for_split))



        if save_to_file:
            self.status = "Started writing details to a text file"
            print(self.status)
            self.plot_grid()
            with open('details.txt', 'a') as f:
                for det in details:
                    f.write("%s\n" % det)
                f.write("\n")
            self.status = "Finished writing details"
            print(self.status)
        else:
            self.status = "Showing grid details:"
            print(self.status)
            for det in details:
                print(det)
            self.status = "Finished printing details"
            print(self.status)

        return None

    def __grid2graph__(self):
        self.grid_graph = nx.Graph()

        for node in self.node_list:
            self.grid_graph.node(node)

        for line in self.line_list:
            self.grid_graph.add_edge(line[0], line[1], arrowhead='none')


        return None


    def plot_grid(self):
        self.__grid2graph__()
        N = nx.nx_agraph.to_agraph(self.grid_graph)
        N.layout()
        N.draw('grid_'+self.grid_id+'.pdf')

        return None