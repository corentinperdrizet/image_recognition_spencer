from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('models/model5.h5')

# Load the label binarizer and its classes
mlb = MultiLabelBinarizer()
mlb.classes_ = ['Complex cultivation patterns', 'Burnt areas', 'Port areas', 'Coastal lagoons', 'Land principally occupied by agriculture, with significant areas of natural vegetation', 
                'Mixed forest', 'Sclerophyllous vegetation', 'Mineral extraction sites', 'Water courses', 'Sparsely vegetated areas', 'Dump sites', 'Industrial or commercial units', 
                'Annual crops associated with permanent crops', 'Intertidal flats', 'Natural grassland', 'Water bodies', 'Continuous urban fabric', 'Rice fields', 
                'Road and rail networks and associated land', 'Olive groves', 'Vineyards', 'Permanently irrigated land', 'Transitional woodland/shrub', 'Pastures', 'Salines', 
                'Broad-leaved forest', 'Agro-forestry areas', 'Peatbogs', 'Bare rock', 'Discontinuous urban fabric', 'Construction sites', 'Coniferous forest', 'Moors and heathland', 
                'Non-irrigated arable land', 'Airports', 'Fruit trees and berry plantations', 'Sport and leisure facilities', 'Inland marshes', 'Green urban areas', 'Sea and ocean', 
                'Salt marshes', 'Estuaries', 'Beaches, dunes, sands']