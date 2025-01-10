from torch import nn
import torch

## basis BERT 'transformers model' + eigen taak layer(s) class
## belangrijk! BERT maakt GEEN word embedings maar contextual embeddings
## TODO uitzoeken 15% sentence mask allows to learn a bidirectional representation of the sentence.
## https://huggingface.co/google-bert/bert-base-uncased
class BERTSwag(nn.Module):

    ## semi constant aantal nodes in het eind van neurale netwerk
    CLASSIFICATION_AMOUNT = 1
    

    def __init__(self, device:torch.device, dropout:float = 0.1):
        ## maak de inherited class (in python 2 moet de class hiërarchie gedefineerd worden in super())
        super().__init__()
        print("Initializing model...")
        # self.device = 
        ##  Arguments inladen als attributes in class
        self.device = device
        ## nn.linear() is een klassieke dense layer en verwacht dus een aant. voor input en output
        ## TODO uitzoeken potentiële invloed bias argument
        ## https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        ## TODO umap reduction
        ## *2 means Linear Transformation (Learned Combination) for 2 embeddings 
        ## umap dim reduction 80%-90%
        self.dense = nn.Sequential(
            nn.Linear(768, 128),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            # nn.Linear(256, 128),
            # nn.LeakyReLU(),
            # nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    ## forward pass met sequence van tokens en attention_mask in tensor/matrix
    ## TODO uitzoeken hoe attention mask wordt gemaakt en geinterpreteerd 
    # def forward(self, input_tokens:torch.Tensor, attention_mask:torch.Tensor):
    def forward(self, resume_embedding, desc_embedding):
        ## BERTS output lijkt hierarchische tensor

        ## Voeg de embeddings samen
        ## TODO uitzoeken cat en andere mogelijkheden
        # combined_embedding = torch.cat((resume_embedding, desc_embedding), dim=1)
        combined_embedding = resume_embedding + desc_embedding
        print(type(combined_embedding))

        ## Other (non linear) ways to test
        # combined_embedding = resume_embedding + desc_embedding
        # combined_embedding = (resume_embedding + desc_embedding) / 2

        ## speciale [CLS] (classification) token, toegevoegd aan het begin van de sequence gebruiken als soort sequence embedding... Andere tokens zijn context embeddings van de woorden in de sentence
        ##TODO tony vragen pooler
        # pooled_output = outputs.pooler_output
        # print("pooled_output:", pooled_output)
        print("combined_embedding: ", combined_embedding)
        return self.dense(combined_embedding)