#g should be a neural net which follows the 3 conditions
#1: invariant to augmentations.
#2: locally lipschitz
#3:similairy between test samples.


#invariant to augmentations ()
def pairwise_similarity(output,tau):
    """the outputs in this case would be aggregated from  2 different augmented inputs"""
    bs=output.shape[0]
    outputs_norm=outputs/(outputs.norm(dim=1).view(bs,1)+1e-8)
    similarity_matrix=(1./tau)*torch.mm(output_norm,outputs_norm.transpose(0,1).detach())
    return (similarity_matrix,outputs)
