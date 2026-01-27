import torch

def get_vertex_connectivity(faces,dtype=torch.int64, device = None):
    '''
    Returns a list of unique edges in the mesh. 
    Each edge contains the indices of the vertices it connects
    返回一个包含mesh中所有边的列表 每条边包含一次
    每条边包含着它所连接的顶点的索引
    '''
    # if torch.is_tensor(faces):
    #     faces = faces.numpy()

    edges = set()
    for f in faces:
        num_vertices = len(f)
        for i in range(num_vertices):
            j = (i + 1) % num_vertices
            edges.add(tuple(sorted([f[i], f[j]])))

    return torch.tensor(list(edges),dtype=dtype, device=device)

# def compute_laplacian_uniform(mesh):
def compute_laplacian_uniform(verts, edges):
    """
    Computes the laplacian in packed form.
    The definition of the laplacian is
    L[i, j] =    -1       , if i == j
    L[i, j] = 1 / deg(i)  , if (i, j) is an edge
    L[i, j] =    0        , otherwise
    where deg(i) is the degree of the i-th vertex in the graph
    Returns:
        Sparse FloatTensor of shape (V, V) where V = sum(V_n)
    """

    # This code is adapted from from PyTorch3D 
    # (https://github.com/facebookresearch/pytorch3d/blob/88f5d790886b26efb9f370fb9e1ea2fa17079d19/pytorch3d/structures/meshes.py#L1128)

    # verts_packed = mesh.vertices # (sum(V_n), 3)
    # edges_packed = mesh.edges    # (sum(E_n), 2)
    verts_packed = verts # (sum(V_n), 3)
    edges_packed = edges   # (sum(E_n), 2)
    V = verts.shape[0]

    e0, e1 = edges_packed.unbind(1)

    idx01 = torch.stack([e0, e1], dim=1)  # (sum(E_n), 2)
    idx10 = torch.stack([e1, e0], dim=1)  # (sum(E_n), 2)
    idx = torch.cat([idx01, idx10], dim=0).t()  # (2, 2*sum(E_n))

    # First, we construct the adjacency matrix,
    # i.e. A[i, j] = 1 if (i,j) is an edge, or
    # A[e0, e1] = 1 &  A[e1, e0] = 1
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=verts.device)
    A = torch.sparse.FloatTensor(idx, ones, (V, V))

    # the sum of i-th row of A gives the degree of the i-th vertex
    deg = torch.sparse.sum(A, dim=1).to_dense()

    # We construct the Laplacian matrix by adding the non diagonal values
    # i.e. L[i, j] = 1 ./ deg(i) if (i, j) is an edge
    deg0 = deg[e0]
    deg0 = torch.where(deg0 > 0.0, 1.0 / deg0, deg0)
    deg1 = deg[e1]
    deg1 = torch.where(deg1 > 0.0, 1.0 / deg1, deg1)
    val = torch.cat([deg0, deg1])
    L = torch.sparse.FloatTensor(idx, val, (V, V))

    # Then we add the diagonal values L[i, i] = -1.
    idx = torch.arange(V, device=verts.device)
    idx = torch.stack([idx, idx], dim=0)
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=verts.device)
    L -= torch.sparse.FloatTensor(idx, ones, (V, V))

    return L

def laplacian_loss(vertices, indices):
    """ Compute the Laplacian term as the mean squared Euclidean norm of the differential coordinates.

    Args:
        mesh (Mesh): Mesh used to build the differential coordinates.
    """

    
    # L = mesh.laplacian
    V = vertices
    # indices = get_vertex_connectivity(faces)
    L = compute_laplacian_uniform(vertices, indices)
    loss = L.mm(V)
    loss = loss.norm(dim=1)**2
    
    return loss.mean()

if __name__ == "__main__":
    import trimesh
    mesh = trimesh.load("tmp/smooth.obj")
    
    verts = torch.from_numpy(mesh.vertices).float()
    faces = torch.from_numpy(mesh.faces)
    print(laplacian_loss(verts,faces))
    
    