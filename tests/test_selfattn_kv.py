self_attention = True
qkv_same_dim = False

assert not self_attention or qkv_same_dim, (
    "Self-attention requires query, key and " "value to be of the same size"
)
