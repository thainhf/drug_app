# double letters for one element turned into single letters that are not in the dataset
double_to_single = {'Si':'q', 'Se':'w', 'Cn':'t', 'Sc':'y', 'Cl':'u', 'Sn':'z', 'Br':'x'} 
single_to_double = {'q':'Si', 'w':'Se', 't':'Cn', 'y':'Sc', 'u':'Cl', 'z':'Sn', 'x':'Br'}
elements_with_double_letters = list(double_to_single)

element_set = ['C','a','(', '=', 'O', ')','/','\\','.','@', 'N', 'c', '1', '$', '2', '3', '4', '#', 'n', 'F', 'u', '-', '[', 'H', ']', 's', 'o', 'S', 't', '5', '6', '+', 'P', 'I', 'x', 'y', 'q', 'B', 'w', '7', '8', 'e', '9', 'b', 'p', '%', '0', 'z']
n_vocab = len(element_set)

element_to_int = dict(zip(element_set, range(0, n_vocab)))
int_to_element = {v: k for k, v in element_to_int.items()}
sequence_length = 100 


smiles_input = layers.Input(shape=(sequence_length,), dtype='int32', name='smiles_input')


embed_smiles = layers.Embedding(output_dim=128, input_dim=n_vocab, input_length=sequence_length)(smiles_input) 


conv1_smiles = tf.compat.v1.keras.layers.CuDNNLSTM(256, return_sequences=True, kernel_initializer=initializers.RandomNormal(stddev=0.2), bias_initializer=initializers.Zeros())(embed_smiles)
activation1_smiles = layers.PReLU()(conv1_smiles)
dropout1 = layers.Dropout(0.1)(activation1_smiles)
conv2_smiles = tf.compat.v1.keras.layers.CuDNNLSTM(512, return_sequences=True, kernel_initializer=initializers.RandomNormal(stddev=0.2), bias_initializer=initializers.Zeros())(dropout1)
activation2_smiles = layers.PReLU()(conv2_smiles)
dropout2 = layers.Dropout(0.1)(activation2_smiles)
conv3_smiles = tf.compat.v1.keras.layers.CuDNNLSTM(256, return_sequences=True, kernel_initializer=initializers.RandomNormal(stddev=0.2), bias_initializer=initializers.Zeros())(dropout2)
activation3_smiles = layers.PReLU()(conv3_smiles)
dropout3 = layers.Dropout(0.1)(activation3_smiles)

# turn into vector
flatten = layers.Flatten()(dropout3)

# dense layers 
dense1 = layers.Dense(512, activation='relu')(flatten)
dropout1_dense = layers.Dropout(0.1)(dense1)
dense2 = layers.Dense(256, activation='relu')(dropout1_dense)

# output
output = layers.Dense(n_vocab, activation="softmax", name='output')(dense2) 

model = Model(inputs=smiles_input, outputs=[output])

model.summary()

model.load_weights('/content/SMILES-best-48.hdf5')

filey = open('/content/pharmaceuticAI_all_compounds.smiles')
structures = [line[:-1] for line in filey]


num_sampled = 1500
random.shuffle(structures)
data = structures[:num_sampled]
data_structs = gen_structs(data)
