import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

class Support_Vector_Machine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1:'r',-1:'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)
    
    # training of svm (convex optimization)
    def fit(self, data):
         self.data = data #training data
         # { ||w||: [w,b] }, convex optimization
         opt_dict = {} #optimization dictionary

         transforms = [[1,1],
                      [-1,1],
                      [-1,-1],
                      [1,-1]] # apply to vector w basically to check every version of vector possible
         #to pick halfway descent starting values for matching with our training data
         all_data = []
         for yi in self.data:
             for featureset in self.data[yi]:
                 for feature in featureset:
                     all_data.append(feature)

         self.max_feature_value = max(all_data)
         self.min_feature_value = min(all_data)
         all_data=None # no need to keep this memory.
         # how to know the optimization is required more is by
         # support vectors yi(xi.w+b) = 1
         
         #in convex optimization problem steps are taken of different sizes to extract the global minima or for minimum value of w vector
         step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      # starts getting very high cost after this.
                      self.max_feature_value * 0.001]
          
          # extremely expensive
         b_range_multiple = 5
         # we don't need to take as small of steps
         # with b as we do with w
         b_multiple = 5
         latest_optimum = self.max_feature_value*10 #first element in vector w
         for step in step_sizes:
             w = np.array([latest_optimum,latest_optimum])
             # initialized optimized var to false we have to reset it in each major step and it will be true when we have checked all steps down to base of our convex shape(bowl's basepoint).
             optimized = False 
             #now to optimize we are iterating through possible 'b' values with constant step size
             while not optimized:

                 for b in np.arange(-1*(self.max_feature_value*b_range_multiple),
                                   self.max_feature_value*b_range_multiple,
                                   step*b_multiple):
                     for transformation in transforms:
                         w_t = w*transformation
                         found_option = True
                         # weakest link in the SVM fundamentally
                         # SMO attempts to fix this a bit
                         # yi(xi.w+b) >= 1
                         # 
                         # #### add a break here later..
                         for i in self.data:
                             for xi in self.data[i]:
                                 yi=i
                                 if not yi*(np.dot(w_t,xi)+b) >= 1:
                                    found_option = False
                                    
                         if found_option:
                             opt_dict[np.linalg.norm(w_t)] = [w_t,b] #mag.of vector w = [w,b] 
                        
                  # as all w are identical        
                    if w[0] < 0:
                        optimized = True
                        print('Optimized a step.')
                    else:
                    # w = [5,5]
                    # step = 1
                    # w - step = [4,4] decreasing w step by step
                         w = w - step
             norms = sorted([n for n in opt_dict]) #sorted list of all the magnitudes of w
             #||w|| : [w,b]
             opt_choice = opt_dict[norms[0]]
             self.w = opt_choice[0]
             self.b = opt_choice[1]
             latest_optimum = opt_choice[0][0]+step*2  #for next step modify the latest_optimum


    def predict(self,features):
        # classifiction is just:
        # sign(xi.w+b)
        classification = np.sign(np.dot(np.array(features),self.w)+self.b)
        # if the classification isn't zero, and we have visualization on, we graph
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0],features[1],s=200,marker='*', c=self.colors[classification])
        else:
            print('featureset',features,'is on the decision boundary')
        return classification

    def visualize(self):
        #scattering known featuresets.
        [[self.ax.scatter(x[0],x[1],s=100,color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

        def hyperplane(x,w,b,v):
            # hyperplane values (v) we are seeking = (w.x+b)
            # positive support vector = +1
            # negattive  = -1
            # db = 0
            return (-w[0]*x-b+v) / w[1]
         # basically we are going to reference variables to draw our hyperplane 
        datarange = (self.min_feature_value*0.9,self.max_feature_value*1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]
        # w.x + b = 1
        # positive sv hyperplane
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min,hyp_x_max], [psv1,psv2], "k")
        # w.x + b = -1
        # negative sv hyperplane
        # k for black color
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min,hyp_x_max], [nsv1,nsv2], "k")

        # w.x + b = 0
        # decision
        # g-- for green color
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min,hyp_x_max], [db1,db2], "g--")

        plt.show()


data_dict = {-1:np.array([[1,7],
                          [2,8],
                          [3,8],]),
             
             1:np.array([[5,1],
                         [6,-1],
                         [7,3],])}