'''
Inception Model for RGB data
'''

def get_model():
    
    from keras.applications.inception_v3 import InceptionV3
    from keras.preprocessing import image
    from keras.models import Model
    from keras.layers import Dense, GlobalAveragePooling2D
    from keras import backend as K
    
    # create the base pre-trained model
    base_model = InceptionV3(weights=None, include_top=False,input_shape=(150, 150, 3))
    
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return(model)
