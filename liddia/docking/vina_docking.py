from typing import Dict
from vina import Vina
from biopandas.pdb import PandasPdb
import os
import AutoDockTools
import argparse
import meeko
import shutil
from hashlib import md5
from uuid import uuid4
from rdkit import Chem
from rdkit.Chem import AllChem

def smiles_to_pdbqt(smiles : str, outdir) -> str:
    """
    This tool will generate 3D coordinates for the given SMILES and save them in a PDBQT file.
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)
    
    preparator = meeko.MoleculePreparation()
    preparator.prepare(mol)
    ligand_pdbqt = preparator.write_pdbqt_string()
    path_to_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(outdir, "ligand.pdbqt"), "w") as f:
        f.write(ligand_pdbqt)

def docking_score(ligand_smiles : str, target_pdb : str, exhaustiveness : int = 8, verbose: int = 1) -> float:
    """
    This function will return the binding affinity score for the given ligand PDBQT and target PDB files.
    The target PDB file can be for an entire protein (blind docking).
    The ligand PDBQT file should contain the 3D coordinates of the ligand.
    Docking will be performed using AutoDock Vina, and the score after docking will be returned.
    """
    path_to_dir = os.path.dirname(os.path.realpath(__file__))
    uuid = str(uuid4().hex)
    os.mkdir(os.path.join(path_to_dir, "datafiles", uuid))
    outdir = os.path.join(path_to_dir, "datafiles", uuid)
    # Convert SMILES to PDBQT
    smiles_to_pdbqt(ligand_smiles, outdir=outdir)

    # Initialize Vina
    v = Vina(sf_name='vina', verbosity=verbose)
    
    # Prepare protein PDBQT
    ppdb = PandasPdb().read_pdb(target_pdb)
    ppdb.df['ATOM'] = ppdb.df['ATOM'][ppdb.df['ATOM']['element_symbol'] != 'H']
    ppdb.to_pdb(path=os.path.join(outdir, "protein_clean.pdb"), records=['ATOM'], gz=False, append_newline=True)
    
    prepare_script = AutoDockTools.__file__.replace("__init__.py", "Utilities24/prepare_receptor4.py")
    os.system(f'python {prepare_script} -r {os.path.join(outdir, "protein_clean.pdb")} -o {os.path.join(outdir, "protein.pdbqt")} -A hydrogens -U waters -U deleteAltB')

    # Set receptor and ligand
    v.set_receptor(os.path.join(outdir, "protein.pdbqt"))
    v.set_ligand_from_file(os.path.join(outdir, "ligand.pdbqt"))

    # Docking: set the search space to cover the entire PDB file
    center = ppdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord']].mean().values
    size = ppdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord']].max().values - ppdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord']].min().values + 10  # Add margin
    
    v.compute_vina_maps(center=center, box_size=size)
    
    # Perform docking
    v.dock(exhaustiveness=exhaustiveness, n_poses=20)    

    # Get docking scores
    scores = v.score()

    shutil.rmtree(outdir) 
    return scores[0]

if __name__ == '__main__':
    # Example usage
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ligand', help='Ligand SMILES', required=True)
    parser.add_argument('--target', help='Path to target PDB file', required=True)
    parser.add_argument('--outfile', help='Path to output file', required=True)
    parser.add_argument('--exhaustiveness', help='Exhaustiveness setting (int > 1)', type=int, default=8)
    args = parser.parse_args()
    score = docking_score(args.ligand, args.target, exhaustiveness=args.exhaustiveness)
    with open(args.outfile, "w") as f:
        f.write(f"Binding affinity score after docking: {score:0.3f}\n")