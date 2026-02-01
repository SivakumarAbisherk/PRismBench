import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim:int, 
                 hidden_dim:tuple=(256, 128, 64), 
                 dropout:tuple=(0.1, 0.1, 0.1), 
                 out_dim:int=4
        ):
        super().__init__()


        layers_stack = []
        n_node_prev = in_dim

        for h, d in zip(hidden_dim, dropout):

            # add linear layer
            layers_stack.append(nn.Linear(n_node_prev, h))

            # add batch normalization layer
            layers_stack.append(nn.BatchNorm1d(h))

            # add ReLU activation
            layers_stack.append(nn.ReLU())
            
            # add dropout
            layers_stack.append(nn.Dropout(d))

            n_node_prev = h

        # output layer
        layers_stack.append(nn.Linear(n_node_prev, out_dim))


        # model architecture
        self.net = nn.Sequential(*layers_stack)
    
    def forward(self, x):
        return self.net(x)
