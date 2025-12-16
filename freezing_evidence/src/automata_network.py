from src.BP import *
from src.population_dynamics import *
from src.population_dynamics_torch import *
from src.graph_BP_regular_torch import *


import numpy as np

import json
import os

homo_messages=[np.array([[1.,0],[0,0]]), np.array([[0., 0],[0,1]]), np.array([[0.5, 0.5],[0,0]]), np.array([[0,0],[0.5,0.5]])]

class automata_network:
    
    def __init__(self, rule, mu=0, cuda=True, BP_reinf=False, N=1000):
        self.rule=rule
        self.mu=mu
        self.BP=BP(rule, mu=mu)
        self.cuda=cuda
        self.population_dynamics=None
        self.BP_reinf=None
        self.N=N
        if cuda:
            self.population_dynamics=population_dynamics_torch(rule, mu=mu)
            if BP_reinf:
                self.BP_reinf=graph_BP_regular_torch(rule=self.rule, N=self.N)
        else:
            self.population_dynamics=population_dynamics(rule, mu=mu)
            if BP_reinf:
                self.BP_reinf=graph_BP_regular_torch(rule=self.rule, N=self.N, device='cpu')
                
        self.cuda=cuda
        self.phase=None
        self.solutions=None
        
    def __repr__(self):
        description='Instance of class \'automata_network\'\n\n'
        if self.phase is not None:
            description+='Phase: '+self.phase
            description+='\nSolution(s): '+self.solutions+'\n\n'
        description+=str(self.BP)+'\n\n'+str(self.population_dynamics)
        if self.BP_reinf is not None:
            description+='\n\n'+str(self.BP_reinf)
        return description
    
    # ============================================================================================================================================
    # Computing RS, d1RSB and s1RSB (using population dynamics and SP) if need and obtain phase and types of solutions.
    # ============================================================================================================================================
    
    def RS_and_1RSB(self,tol=1e-3, tol_BP=1e-10, tol_message_comparison=0.99999, compute_SP=True):
        if self.cuda==True:
            self.BP.find_all_fixed_points_torch()
        else:
            self.BP.find_all_fixed_points()

        self.BP.is_stable_AT()
        self.BP.analytic_stability()
        if compute_SP:
            self.BP.run_SP()
            self.BP.is_stable_SP()

        if self.cuda:
            self.population_dynamics=population_dynamics_torch(self.rule, mu=self.mu, BP_message=self.BP.psi)
        else:
            self.population_dynamics=population_dynamics(self.rule, mu=self.mu, BP_message=self.BP.psi)
            
        self.population_dynamics.run()
        self.population_dynamics.stability_check()
                
        number_homogeneous_configurations=0
        if self.rule[0]=='0' or self.rule[0]=='+':
            number_homogeneous_configurations+=1
        if self.rule[-1]=='1' or self.rule[-1]=='+':
            number_homogeneous_configurations+=1
        
        isnan=False
        if self.BP.physical:
            if np.isnan(self.population_dynamics.phi_mean) or np.isneginf(self.population_dynamics.phi_mean):
                print("Should not be here 1")
                isnan=True
        RS_same_message_1RSB=False
        pop=self.population_dynamics.population.cpu().numpy()
        for idx, fixed_point in enumerate(self.BP.fixed_points):
            if np.sum(np.linalg.norm(fixed_point-pop, axis=(1,2))<tol)/self.population_dynamics.M>tol_message_comparison:
                RS_same_message_1RSB=True
                break

        num_homo=0
        num_phi_0=0
        for j in range(len(self.BP.all_phi)):
            if abs(self.BP.all_phi[j])<tol_BP:
                num_phi_0+=1
            if np.any(np.all(np.round(self.BP.fixed_points[j],1) == homo_messages, axis=(1,2))):
                num_homo+=1
                
        if isnan or self.BP.physical==False or RS_same_message_1RSB:
            self.phase='RS'
            if self.BP.phi<-5:
                self.solutions='locally contradictory'
            elif self.BP.phi<-tol_BP:
                self.solutions='No stationary configuration but not locally contradictory'
            elif np.abs(self.BP.phi)<tol_BP:
                
                if num_homo!=number_homogeneous_configurations:
                    print("Some homogeneous configuration was not found by BP for rule ", self.rule)
                if num_homo==num_phi_0:
                    self.solutions='Only homogeneous stationary solutions'
                else:
                    if num_phi_0>num_homo and num_homo>0:
                        self.solutions='Subexponentially many with homogeneous stationary solutions'
                    elif num_phi_0>num_homo and num_homo==0:
                        self.solutions='Subexponentially many with no homogeneous stationary solutions'
                    else:
                        print(num_phi_0)
                        print(num_homo)
                        print("Should not be here 2")
            elif self.BP.phi>tol_BP:
                if number_homogeneous_configurations>0:
                    self.solutions="Exponentially many with homogeneous stationary solutions"
                else:
                    self.solutions="Exponentially many with no homogeneous stationary solutions"

        
        elif self.BP.phi+tol>=self.population_dynamics.phi_mean:
            if self.population_dynamics.complexity>-tol:
                if self.population_dynamics.fraction_i_frozen<tol:
                    self.phase='d1RSB'
                elif self.population_dynamics.fraction_i_frozen<1-tol:
                    self.phase='r1RSB'
                else:
                    self.phase='l1RSB'
                    
                if self.population_dynamics.psi_mean<-5:
                    self.solutions='locally contradictory'
                if self.population_dynamics.psi_mean<-tol:
                    self.solutions='No stationary configuration but not locally contradictory'
                elif np.abs(self.population_dynamics.psi_mean)<tol:
                    if num_homo==0:
                        self.solutions='Subexponentially many with no homogeneous stationary solutions'
                    else:
                        self.solutions='Subexponentially many with homogeneous stationary solutions'

                elif self.population_dynamics.psi_mean>tol:
                    if number_homogeneous_configurations>0:
                        self.solutions="Exponentially many with homogeneous stationary solutions"
                    else:
                        self.solutions="Exponentially many with no homogeneous stationary solutions"
                        
            else:
                self.phase='s1RSB'
                self.population_dynamics.compute_complexity_curves()
                if self.population_dynamics.phi_s<-5:
                    self.solutions='locally contradictory'
                elif self.population_dynamics.phi_s<-tol:
                    self.solutions='No stationary configuration but not locally contradictory'
                elif np.abs(self.population_dynamics.phi_s)<tol:
                    if num_homo==0:
                        self.solutions='Subexponentially many with no homogeneous stationary solutions'
                    else:
                        self.solutions='Subexponentially many with homogeneous stationary solutions'
                elif self.population_dynamics.phi_s>tol:
                    if number_homogeneous_configurations>0:
                        self.solutions="Exponentially many with homogeneous stationary solutions"
                    else:
                        self.solutions="Exponentially many with no homogeneous stationary solutions"
                        
                if not any(self.population_dynamics.complexity_list>0):
                    self.phase='NPC'

            
        elif self.BP.phi+tol<self.population_dynamics.phi_mean:
            print("Warning... BP free entropy smaller than population dynamics free entropy for rule ",self.rule, ", computing static 1RSB and hoping that this solves the problem.")
            self.phase='s1RSB'
            self.population_dynamics.compute_complexity_curves()
            if self.population_dynamics.phi_s>self.BP.phi+tol and any(self.population_dynamics.complexity_list>0):
                print("Warning: the BP free entropy is still smaller that the population dynamics free entropy even after static 1RSB calculation !")
            else:
                print("Problem solved with static 1RSB, all good !")
            if not any(self.population_dynamics.complexity_list>0):
                self.phase='NPC'
                if self.population_dynamics.phi_s<-5:
                    self.solutions='Locally contradictory'
                else:
                    self.solutions='No stationary configuration but not locally contradictory'
            elif self.population_dynamics.phi_s<-5:
                self.solutions='Locally contradictory'
            elif self.population_dynamics.phi_s<-tol:
                self.solutions='No stationary configuration but not locally contradictory'
            elif np.abs(self.population_dynamics.phi_s)<=tol:
                self.solutions='Subexponentially many stationary solutions, no informations on homogeneous'
            elif self.population_dynamics.phi_s>tol:
                    if number_homogeneous_configurations>0:
                        self.solutions="Exponentially many with homogeneous stationary solutions"
                    else:
                        self.solutions="Exponentially many with no homogeneous stationary solutions"
        else:
            print("Should not be here 3")

    # ============================================================================================================================================
    # Run multiple experiments to see if we find solutions
    # ============================================================================================================================================

    def BP_reinforcement(self, num_runs=10, Ns=[1000,10000,100000], verbose=0):
        
        if self.BP_reinf is None:
            self.BP_reinf=graph_BP_regular_torch(rule=self.rule, N=1000)
            
        self.BP_reinf.multiple_runs(num_runs=num_runs, Ns=Ns, verbose=verbose)

    # ============================================================================================================================================
    # Saving and loading
    # ============================================================================================================================================
    
    def save(self, folder_automata_network=None, folder_BP=None, folder_population_dynamics=None, folder_BP_reinf=None, save_population=False):
        self.BP.save(folder=folder_BP)
        self.population_dynamics.save(folder=folder_population_dynamics, save_population=save_population)
        if self.BP_reinf is not None:
            self.BP_reinf.save(folder_BP_reinf)

        if folder_automata_network is None:
            folder_automata_network='results/'+str(self.BP.rule)+'/automata_network_mu='+str(self.BP.mu)
        if not os.path.exists(folder_automata_network):
            os.makedirs(folder_automata_network)
        self.save_automata_network(folder_automata_network+'/automata_network.json')
    
    def save_automata_network(self, path=None):
        if path is None:
            path='results/'+str(self.rule)+'/automata_network_mu='+str(self.BP.mu)
            if not os.path.exists(path):
                os.makedirs(path)
            path+='/automata_network.json'
        
        phase={'rule': self.rule,  'phase': self.phase, 'solutions':self.solutions}
        with open(path, 'w+') as json_file:
            json.dump(phase, json_file)
            
    def load_automata_network(self, path=None):
        if path is None:
            path='results/'+str(self.BP.rule)+'/automata_network_mu='+str(self.BP.mu)+'/automata_network.json'
        if os.path.exists(path):
            dic=json.load(open(path))
            self.rule=dic['rule']
            self.phase=dic['phase']
            self.solutions=dic['solutions']
            
    def load(self, load_population=False, load_BP_reinf=False, load_BP_reinf_graph=False, path_automata_network=None, folder_BP=None, folder_population_dynamics=None):
        if path_automata_network is None:
            path_automata_network='results/'+str(self.rule)+'/automata_network_mu='+str(self.BP.mu)+'/automata_network.json'
        if folder_BP is None:
            folder_BP='results/'+str(self.rule)+'/BP_mu='+str(self.BP.mu)
        if folder_population_dynamics is None:
            folder_population_dynamics='results/'+str(self.rule)+'/population_dynamics_mu='+str(self.BP.mu)
            
        self.BP.load(folder=folder_BP)
        self.population_dynamics.load(folder=folder_population_dynamics, load_population=load_population)
        if load_BP_reinf:
            self.BP_reinf=graph_BP_regular_torch(rule=self.rule, N=100)
            self.BP_reinf.load(load_messages_and_graph=load_BP_reinf_graph)
            
        self.load_automata_network(path=path_automata_network)
            
            
            
            

