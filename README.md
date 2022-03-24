# Hyperspectral-Classification-framework
This is a light framework for patch-based hyperspectral classification by pytorch. This framework builds a pipline of HSC, including hyperspectral data loading (**Pavia University, IndianPines and Salinas scene** datasets are pre-defined), patch inputs generation, personal neural network establishment, training, valuating, inferring and evaluating.

# Usage
1. Add you data to **.\dataset\\** and load your data in *loadData* function.
``` python
def loadData(name): ## customize data and return data label and class_name
    data_path = os.path.join(os.getcwd(),'dataset')
    if name == 'IP':
        data = loadmat(os.path.join(data_path, 'IndianPines\\Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = loadmat(os.path.join(data_path, 'IndianPines\\Indian_pines_gt.mat'))['indian_pines_gt']
        class_name = [ "Alfalfa", "Corn-notill", "Corn-mintill","Corn", "Grass-pasture", 
                       "Grass-trees","Grass-pasture-mowed", "Hay-windrowed", "Oats","Soybean-notill", "Soybean-mintill", "Soybean-clean","Wheat", "Woods", "Buildings-Grass-Trees-Drives","Stone-Steel-Towers"]
```
2. Bulid your model and test.
```python
class MODEL(nn.Module):

    def __init__(self, input_channels, n_classes, *args):
        super(MODEL, self).__init__()
  
    def forward(self, x):
        return x
net = MODEL(......)
summary(net, ......)
```
3. Set your experimental paremeters.
```python
##hypeperameters and experimental settings
RANDOM_SEED=666
MODEL_NAME = 'CNN1D' ## your model name
DATASET = 'PU'  ## PU  IP  SA or your personal dataset
TRAIN_RATE = 0.1  ## ratio of training data
VAL_RATE = 0.05    ## ratio of valuating data
EPOCH = 100    ##number of epoch
VAL_EPOCH = 1  ##interval of valuation
LR = 0.001    ##learning rate
WEIGHT_DECAY = 1e-6  
BATCH_SIZE = 64
DEVICE = 0  ##-1:CPU  0:cuda 0
N_PCA = 15  ## reserved PCA components   0:use origin data
NORM = True  ## normalization or not
PATCH_SIZE = 1 ## patchsize of input 3D cube   1:only spectral sequence
CHECK_POINT = None  ## path of check point model
```
4. Run the Code ~~and Debug~~.

# Requirements

pytorch 1.9.0

scikit-learn 1.0.2

spectral 0.22.2

torchinfo 1.6.1
