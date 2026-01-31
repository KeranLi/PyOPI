"""
Solutions File Handler for OPI

Handles reading and writing of OPI optimization solution files.
Matches MATLAB's getSolutions.m and writeSolutions.m
"""

import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, TextIO
import numpy as np


class SolutionsFileWriter:
    """
    Writer for OPI solutions files.
    
    Matches MATLAB's writeSolutions function behavior.
    """
    
    def __init__(self):
        self.fid: Optional[TextIO] = None
        self.is_started = False
        self.n_iterations = 0
        self.n_parameters = 0
        self.filepath = None
    
    def initialize(self, run_path: str, run_title: str, n_samples: int,
                   parameter_labels: List[str], exponents: List[int],
                   lb: List[float], ub: List[float]) -> None:
        """
        Initialize solutions file.
        
        Parameters
        ----------
        run_path : str
            Directory for solutions file
        run_title : str
            Run title
        n_samples : int
            Number of observations
        parameter_labels : list
            Parameter label strings
        exponents : list
            Power-of-10 scaling factors
        lb, ub : list
            Lower and upper bounds
        """
        self.n_parameters = len(exponents)
        self.filepath = os.path.join(run_path, 'opiFit_Solutions.txt')
        
        # Backup existing file
        if os.path.exists(self.filepath):
            backup_path = os.path.join(run_path, 'opiFit_Solutions.bk!')
            try:
                os.rename(self.filepath, backup_path)
            except:
                pass
        
        self.fid = open(self.filepath, 'w', encoding='utf-8')
        self.is_started = True
        self.n_iterations = 0
        
        # Write header
        self.fid.write(f"% Solution file for optimization using opiFit\n")
        self.fid.write(f"% Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.fid.write(f"% Original run path:\n% {run_path}\n")
        self.fid.write(f"% Run title:\n{run_title}\n")
        self.fid.write(f"% Number of observations:\n{n_samples}\n")
        self.fid.write(f"% Parameter labels:\n")
        self.fid.write("|".join(parameter_labels) + "\n")
        self.fid.write(f"% Exponent for power of 10 factoring for plotted variables labeled above:\n")
        self.fid.write("\t".join(str(e) for e in exponents) + "\n")
        self.fid.write(f"% Lower and upper constraints for parameter search:\n")
        self.fid.write("\t".join(str(b) for b in lb) + "\n")
        self.fid.write("\t".join(str(b) for b in ub) + "\n")
    
    def write_solution(self, iteration: int, chi_r2: float, nu: int, 
                       beta: List[float]) -> None:
        """
        Write a solution to the file.
        
        Parameters
        ----------
        iteration : int
            Iteration number
        chi_r2 : float
            Reduced chi-square value
        nu : int
            Degrees of freedom
        beta : list
            Parameter vector
        """
        if not self.is_started or self.fid is None:
            raise RuntimeError("Solutions file not initialized")
        
        # Write column headers on first call
        if self.n_iterations == 0:
            self.fid.write(f"%    n\t        chiR2\t    nu")
            for i in range(1, self.n_parameters + 1):
                self.fid.write(f"\t         beta{i:02d}")
            self.fid.write("\n")
        
        self.n_iterations += 1
        
        # Format: iteration, chi_r2, nu, beta values
        line = f"{iteration:6d}\t{chi_r2:15.6g}\t{nu:6d}"
        for b in beta:
            line += f"\t{b:15.6g}"
        line += "\n"
        
        self.fid.write(line)
        self.fid.flush()  # Ensure immediate write
    
    def write_note(self, text: str) -> None:
        """Write a comment/note to the file."""
        if self.fid is not None:
            self.fid.write(f"% {text}\n")
    
    def close(self) -> None:
        """Close the solutions file."""
        if self.is_started and self.fid is not None:
            self.fid.write(f"% Finish time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.fid.close()
            self.is_started = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def parse_solutions_file(solutions_path: str, solutions_file: str) -> Dict[str, Any]:
    """
    Parse an OPI solutions file.
    
    Matches MATLAB's getSolutions function.
    
    Parameters
    ----------
    solutions_path : str
        Directory containing solutions file
    solutions_file : str
        Solutions filename
    
    Returns
    -------
    results : dict
        Dictionary containing:
        - run_title: run title string
        - n_samples: number of observations
        - axis_labels: parameter labels
        - exponents: power-of-10 scaling factors
        - lb_all: lower bounds
        - ub_all: upper bounds
        - m_set: search set size
        - epsilon0: stopping criterion
        - solutions_all: array of all solutions (n_solutions x (3+n_parameters))
        - chi_r2_best: best chi-square value
        - nu_best: degrees of freedom for best solution
        - beta_best: best parameter vector
    """
    filepath = os.path.join(solutions_path, solutions_file)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Solutions file not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Process lines - skip comments and empty lines
    content_lines = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('%'):
            # Strip null characters
            line = line.strip('\x00')
            content_lines.append(line)
    
    idx = 0
    
    # Run title
    run_title = content_lines[idx]; idx += 1
    
    # Number of samples
    n_samples = int(float(content_lines[idx])); idx += 1
    
    # Parameter labels
    axis_labels_all = [label.strip() for label in content_lines[idx].split('|')]; idx += 1
    n_parameters = len(axis_labels_all)
    
    # Exponents
    exponents_str = content_lines[idx]; idx += 1
    exponents_all = [int(x) for x in re.findall(r'-?\d+', exponents_str)]
    
    # Lower bounds
    lb_str = content_lines[idx]; idx += 1
    lb_all = [float(x) for x in re.findall(r'-?\d+\.?\d*(?:e[+-]?\d+)?', lb_str)]
    
    # Upper bounds
    ub_str = content_lines[idx]; idx += 1
    ub_all = [float(x) for x in re.findall(r'-?\d+\.?\d*(?:e[+-]?\d+)?', ub_str)]
    
    # Search set size (m_set)
    m_set = int(float(content_lines[idx])); idx += 1
    
    # Epsilon0
    epsilon0 = float(content_lines[idx]); idx += 1
    
    # Read solutions until 'END' or end of file
    solutions_all = []
    while idx < len(content_lines):
        line = content_lines[idx]
        idx += 1
        
        if line.upper().startswith('END'):
            break
        
        # Parse solution: iteration, chi_r2, nu, beta values
        values = [float(x) for x in re.findall(r'-?\d+\.?\d*(?:e[+-]?\d+)?', line)]
        
        # Check for complete solution (iteration + chi_r2 + nu + n_parameters)
        if len(values) == n_parameters + 3 or len(values) == n_parameters + 4:
            # Skip iteration number (first value)
            solutions_all.append(values[1:])
    
    if not solutions_all:
        raise ValueError("No valid solutions found in file")
    
    solutions_all = np.array(solutions_all)
    
    # Find best solution (minimum chi_r2)
    i_min = np.argmin(solutions_all[:, 0])
    chi_r2_best = solutions_all[i_min, 0]
    nu_best = int(solutions_all[i_min, 1])
    beta_best = solutions_all[i_min, 2:]
    
    return {
        'run_title': run_title,
        'n_samples': n_samples,
        'axis_labels': axis_labels_all,
        'exponents': exponents_all,
        'lb_all': lb_all,
        'ub_all': ub_all,
        'm_set': m_set,
        'epsilon0': epsilon0,
        'solutions_all': solutions_all,
        'chi_r2_best': chi_r2_best,
        'nu_best': nu_best,
        'beta_best': beta_best,
        'n_parameters': n_parameters,
        'n_solutions': len(solutions_all),
    }


def get_best_solution(solutions_path: str, solutions_file: str) -> Dict[str, Any]:
    """
    Get only the best solution from a solutions file.
    
    Parameters
    ----------
    solutions_path : str
        Directory containing solutions file
    solutions_file : str
        Solutions filename
    
    Returns
    -------
    best_solution : dict
        Best solution with beta, chi_r2, nu
    """
    results = parse_solutions_file(solutions_path, solutions_file)
    
    return {
        'beta': results['beta_best'],
        'chi_r2': results['chi_r2_best'],
        'nu': results['nu_best'],
        'run_title': results['run_title'],
        'n_samples': results['n_samples'],
    }


def merge_solutions_files(file_list: List[Tuple[str, str]], 
                          output_path: str) -> Dict[str, Any]:
    """
    Merge multiple solutions files into one.
    
    Parameters
    ----------
    file_list : list
        List of (path, filename) tuples
    output_path : str
        Output file path
    
    Returns
    -------
    merged_info : dict
        Information about merged solutions
    """
    all_solutions = []
    header_info = None
    
    for path, filename in file_list:
        results = parse_solutions_file(path, filename)
        
        if header_info is None:
            header_info = {
                'run_title': results['run_title'] + ' (merged)',
                'n_samples': results['n_samples'],
                'axis_labels': results['axis_labels'],
                'exponents': results['exponents'],
                'lb_all': results['lb_all'],
                'ub_all': results['ub_all'],
                'm_set': results['m_set'],
                'epsilon0': results['epsilon0'],
            }
        
        all_solutions.extend(results['solutions_all'])
    
    # Write merged file
    with SolutionsFileWriter() as writer:
        writer.initialize(
            run_path=os.path.dirname(output_path),
            run_title=header_info['run_title'],
            n_samples=header_info['n_samples'],
            parameter_labels=header_info['axis_labels'],
            exponents=header_info['exponents'],
            lb=header_info['lb_all'],
            ub=header_info['ub_all']
        )
        
        for i, sol in enumerate(all_solutions):
            writer.write_solution(
                iteration=i+1,
                chi_r2=sol[0],
                nu=int(sol[1]),
                beta=sol[2:].tolist()
            )
    
    return {
        'n_solutions': len(all_solutions),
        'output_path': output_path,
    }
