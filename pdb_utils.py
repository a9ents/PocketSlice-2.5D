import numpy as np
import warnings
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning

warnings.filterwarnings("ignore", category=PDBConstructionWarning)
WATER_NAMES = {"HOH", "WAT", "DOD", "H2O"}

def load_pdb(pdb_path, ligand_resname, chain_id=None):
    """
    返回: 
      protein_coords (N,3)
      protein_chem (N,)  # 化学特征: C/S = 1.0, O/N = -1.0, 其他 = 0.0
      ligand_coords (M,3)
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", pdb_path)

    protein_atoms, protein_chem, ligand_atoms = [], [], []
    
    for model in structure:
        for chain in model:
            if chain_id is not None and chain.id != chain_id:
                continue
            for residue in chain:
                if residue.id[0] == " " and residue.resname not in WATER_NAMES:
                    for atom in residue:
                        protein_atoms.append(atom.coord)
                        elem = atom.element.upper()
                        # 碳/硫 -> 疏水(1); 氧/氮 -> 亲水(-1); 其他 -> 0
                        if elem in ["C", "S"]:
                            protein_chem.append(1.0)
                        elif elem in ["O", "N"]:
                            protein_chem.append(-1.0)
                        else:
                            protein_chem.append(0.0)
                elif residue.resname == ligand_resname:
                    for atom in residue:
                        ligand_atoms.append(atom.coord)
        break 

    protein_coords = np.array(protein_atoms, dtype=np.float32)
    protein_chem = np.array(protein_chem, dtype=np.float32)
    ligand_coords = np.array(ligand_atoms, dtype=np.float32)
    
    return protein_coords, protein_chem, ligand_coords

# 注意：pca_axes 和 ligand_centroid 保持不变，保留在文件末尾即可

def pca_axes(coords):
    """返回 3x3 矩阵，列为主轴（按方差降序）。"""
    centered = coords - coords.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    return eigvecs[:, idx].astype(np.float32)


def ligand_centroid(ligand_coords):
    return ligand_coords.mean(axis=0)