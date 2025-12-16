import numpy as np
import networkx as nx
import itertools
from utils.rules import *
from utils.graph import *

import torch

import json
import csv
import ast
import os

class graph_BP_regular_torch:
    def __init__(self, rule, edges=None, N=100000, mu=0, kappa=0, tol=1e-12, max_iter=10000, damping_parameter=0.8, init_messages=None, hard_fields=False, pi=0.25, gamma=0.1, seed=None, device='cuda'):
        self.rule=rule
        self.mu=mu
        self.kappa=kappa
        self.tol=tol
        self.max_iter=max_iter
        self.damping_parameter=damping_parameter
        self.hard_fields=hard_fields
        self.pi=pi
        self.gamma=gamma
        self.rng=np.random.default_rng()
        if seed is not None:
            self.rng=np.random.default_rng(seed)
            torch.manual_seed(seed)
        self.device=device
        #if self.device=='cuda': # Depending on the version of torch this should be used
        #    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
        torch.set_default_device('cuda')
        torch.set_default_dtype(torch.double)
            

        self.d=len(self.rule)-1
        
        self.N=N
        if edges is None:
            edges=torch.tensor(list(random_regular_graph_edges(d=self.d,n=self.N)))

        self.num_edges=len(edges)
        self.M=2*self.num_edges
        self.N=int(self.M/(self.d))
        self.edges=torch.cat((edges, torch.flip(edges, dims=(1,0))))
        
        self.truncated_connectivity=torch.zeros((self.M,self.d-1),dtype=torch.long)
        for idx,edge in enumerate(self.edges):
            self.truncated_connectivity[idx]=torch.argwhere(torch.logical_and(self.edges[:,1]==edge[0],self.edges[:,0]!=edge[1])).flatten()
        
        self.connectivity=torch.zeros((self.N, self.d), dtype=torch.long)
        for i in range(self.N):
            self.connectivity[i,:]=torch.argwhere(self.edges[:,1]==i).flatten()
                
        self.truncated_node_connectivity=self.edges[self.truncated_connectivity][:,:,0]
        self.node_connectivity=self.edges[self.connectivity][:,:,0]
        self.config=None
                           
        
        self.psi=init_messages
        if hard_fields==True:
            self.psi=self.rng.choice(np.array([[[1.,0],[0,0]], [[0.,1],[0,0]], [[0.,0],[1,0]], [[0.,0],[0,1]]], dtype=np.float32), self.M)
            self.psi=torch.from_numpy(self.psi).to(self.device).double()
        elif self.psi is None:
            self.psi=torch.rand(size=(self.M,2,2))
            self.psi=self.psi/self.psi.sum(dim=(1,2)).repeat_interleave(4).reshape(self.M,2,2).double()
        
        self.psi_old=self.psi.clone()
        
        self.psi_reinforced=self.psi.clone()
        self.psi_old_reinforced=self.psi.clone()
        self.rho_site=None
        self.marginals_reinforced=None
        self.site_marginals_reinforced=None

        self.unique_indexes=torch.from_numpy(np.unique(self.edges[:,0].cpu().numpy(), return_index=True)[1])

       
        
            
        self.permutations=np.array(list(itertools.product([0,1], repeat=self.d-1)))
        self.permutations=torch.tensor(self.permutations)
        
        self.phi=None
        self.rho=None
        self.distance=None
        self.s=None
        self.local_entropy=None
        self.marginals=None

        self.stability=None
        self.stability_non_zero=None

        self.fraction_hard_fields=None # message has a 1 somewhere (and thus the rest is 0)
        self.fraction_i_frozen=None # i is uniquely determined, i.e. the site marginals is either exactly 0 or 1
        self.fraction_config_forbidden=None # the message contains a 0 --> a configuration is forbidden, either for i or j

        # Results of multiple tries to find a solution
        self.fraction_found=None
        
        
        
             
        
    def __repr__(self):
        description="Instance of class \'graph_BP_regular_torch\'\nRule : "+str(self.rule)
        description+='\nN = '+str(self.N)+"\nμ =  "+str(self.mu)
        if self.phi is not None:
            description+='\nφ = '+str(self.phi)
            description+='\nρ = '+str(self.rho)
            description+='\ns = '+str(self.s)
        if self.stability is not None:
            description+='\nStability : '+str(self.stability)
            description+='\nStability noise only non-zero : '+str(self.stability_non_zero)
        if self.fraction_hard_fields is not None:
            description+='\n\nFraction hard fields : '+ str(self.fraction_hard_fields)
            description+='\nFraction i frozen : '+ str(self.fraction_i_frozen)
            description+='\nFraction at least one config forbidden : '+ str(self.fraction_config_forbidden)
        if self.fraction_found is not None:
            description+='\nFraction found : '+str(self.fraction_found)

        return description
    
    #=============================================================================================================================
    # Belief propagation and observables
    #============================================================================================================================= 

    def reset_messages(self):
        self.psi=None
        if self.hard_fields==True:
            self.psi=self.rng.choice(np.array([[[1.,0],[0,0]], [[0.,1],[0,0]], [[0.,0],[1,0]], [[0.,0],[0,1]]], dtype=np.float32), self.M)
            self.psi=torch.from_numpy(self.psi).to(self.device).double()
        elif self.psi is None:
            self.psi=torch.rand(size=(self.M,2,2))
            self.psi=self.psi/self.psi.sum(dim=(1,2)).repeat_interleave(4).reshape(self.M,2,2).double()
        
        self.psi_old=self.psi.clone()
        
        self.psi_reinforced=self.psi.clone()
        self.psi_old_reinforced=self.psi.clone()
    

    def respect_rule(self, i, j, rest_config):
        outer_density=j+torch.sum(rest_config)
        if self.rule[outer_density]=='0':
            return True if i==0 else False
        elif self.rule[outer_density]=='1':
            return True if i==1 else False
        elif self.rule[outer_density]=='+':
            return True
        elif self.rule[outer_density]=='-':
            return False
    
    def respect_rule_v2(self, i, outer_density):
        if self.rule[outer_density]=='0':
            return True if i==0 else False
        elif self.rule[outer_density]=='1':
            return True if i==1 else False
        elif self.rule[outer_density]=='+':
            return True
        elif self.rule[outer_density]=='-':
            return False
                        
                        
    def diff(self):
        return torch.mean(torch.linalg.norm(self.psi-self.psi_old, dim=(1,2)))
    
    
    def step_regular(self):
        self.psi_old=self.psi.clone()
        self.psi=torch.zeros((self.M,2,2))
        for i in range(2):
            for j in range(2):
                for perm in self.permutations:
                    mult=torch.ones(self.M)
                    for k in range(self.d-1):
                        mult*=self.psi_old[self.truncated_connectivity[:,k]][:,perm[k],i]

                    self.psi[:,i,j]+=np.exp(self.mu*i)*self.respect_rule(i, j, perm)*mult

        Z=torch.sum(self.psi, dim=(1,2))
        self.psi[torch.where(Z==0)]=torch.tensor([[0.25,0.25],[0.25,0.25]])
        Z=torch.where(Z!=0.,Z,1.)
        self.psi/=Z.repeat_interleave(4).reshape(self.M,2,2)
        self.psi=self.damping_parameter*self.psi_old+(1-self.damping_parameter)*self.psi
        
 
    def update_observables(self):
        Z_i=torch.zeros(self.N)
        numerator=torch.zeros(self.N)
        for i in range(2):
            for j in range(2):
                for perm in self.permutations:
                    mult=self.psi[self.connectivity[:,0]][:,j,i]
                    for k in range(self.d-1):
                        mult*=self.psi[self.connectivity[:,k+1]][:,perm[k],i]
                    Z_i+=np.exp(self.mu*i)*self.respect_rule(i,j,perm)*mult
                    numerator+=i*np.exp(self.mu*i)*self.respect_rule(i,j,perm)*mult
        phi=torch.sum(torch.log(Z_i))-torch.sum(torch.log(torch.sum(self.psi[:self.num_edges]*torch.flip(torch.transpose(self.psi[self.num_edges:],1,2),dims=(0,)),dim=(1,2))))
        rho=torch.sum(numerator/Z_i)
        
        self.phi=(phi/self.N).item()
        self.rho=(rho/self.N).item()
        self.s=self.phi-self.mu*self.rho
        

    def update_marginals(self):
        self.marginals=torch.zeros((self.M,2,2))
        self.marginals[:,0,0]=self.psi[:,0,0]*torch.flip(self.psi,dims=(0,))[:,0,0]
        self.marginals[:,1,0]=self.psi[:,1,0]*torch.flip(self.psi,dims=(0,))[:,0,1]
        self.marginals[:,0,1]=self.psi[:,0,1]*torch.flip(self.psi,dims=(0,))[:,1,0]
        self.marginals[:,1,1]=self.psi[:,1,1]*torch.flip(self.psi,dims=(0,))[:,1,1]
        Z=torch.sum(self.marginals, dim=(1,2))
        self.marginals[torch.where(Z==0.)]=torch.tensor([[0.25,0.25],[0.25,0.25]])
        Z=torch.where(Z!=0.,Z,1.)
        self.marginals/=Z.repeat_interleave(4).reshape(self.M,2,2)
    
    def run(self,max_iter=None, tol=None, verbose=False):
        if max_iter is None:
            max_iter=self.max_iter
        if tol is None:
            tol=self.tol
        for i in range(max_iter):
            self.step_regular()
            diff=self.diff()
            if verbose>=2 and i%int(max_iter/20)==0:
                print('Iter : ', i+1, 'Diff = ', diff)
            if diff<tol:
                break
        if i==max_iter-1:
            print('No convergence reached for rule ', self.rule, 'after ', i,' steps. Final error: ', self.diff())
        else:
            if verbose:
                print('Converged after ', i, ' steps. Final error: ', self.diff())
        self.update_observables()
        self.update_marginals()
        self.compute_frozen()
        
     

    #=============================================================================================================================
    # Belief propagation Reinforcement
    #=============================================================================================================================

    def reset_graph(self, edges=None, N=None):
        if N is not None:
            self.N=N
        if edges is None:
            edges=torch.tensor(list(random_regular_graph_edges(d=self.d,n=self.N)))
            self.edges=torch.cat((edges, torch.flip(edges, dims=(1,0))))
        else:
            self.edges=edges

        self.num_edges=int(len(self.edges)/2)
        self.M=len(self.edges)
        self.N=int(self.M/(self.d))
        
        self.truncated_connectivity=torch.zeros((self.M,self.d-1),dtype=torch.long)
        for idx,edge in enumerate(self.edges):
            self.truncated_connectivity[idx]=torch.argwhere(torch.logical_and(self.edges[:,1]==edge[0],self.edges[:,0]!=edge[1])).flatten()
        
        self.connectivity=torch.zeros((self.N, self.d), dtype=torch.long)
        for i in range(self.N):
            self.connectivity[i,:]=torch.argwhere(self.edges[:,1]==i).flatten()
                
        self.truncated_node_connectivity=self.edges[self.truncated_connectivity][:,:,0]
        self.node_connectivity=self.edges[self.connectivity][:,:,0]
        self.config=None

        self.unique_indexes=torch.from_numpy(np.unique(self.edges[:,0].cpu().numpy(), return_index=True)[1])


    def diff_reinforced(self):
        return torch.mean(torch.linalg.norm(self.psi_reinforced-self.psi_old_reinforced, dim=(1,2)))
    

    def update_marginals_reinforced(self):
        self.marginals_reinforced=torch.zeros((self.M,2,2))
        self.marginals_reinforced[:,0,0]=self.psi_reinforced[:,0,0]*torch.flip(self.psi_reinforced,dims=(0,))[:,0,0]
        self.marginals_reinforced[:,1,0]=self.psi_reinforced[:,1,0]*torch.flip(self.psi_reinforced,dims=(0,))[:,0,1]
        self.marginals_reinforced[:,0,1]=self.psi_reinforced[:,0,1]*torch.flip(self.psi_reinforced,dims=(0,))[:,1,0]
        self.marginals_reinforced[:,1,1]=self.psi_reinforced[:,1,1]*torch.flip(self.psi_reinforced,dims=(0,))[:,1,1]
        Z=torch.sum(self.marginals_reinforced, dim=(1,2))
        self.marginals_reinforced[torch.where(Z==0.)]=torch.tensor([[0.25,0.25],[0.25,0.25]])
        Z=torch.where(Z!=0.,Z,1.)
        self.marginals_reinforced/=Z.repeat_interleave(4).reshape(self.M,2,2)
        

    def update_site_marginals(self):
        self.update_marginals_reinforced()
        x0=1
        x1=1
        for k in range(self.d):
            x0*=(self.marginals_reinforced[self.connectivity[:,k],0,0]+self.marginals_reinforced[self.connectivity[:,k],1,0])
            x1*=(self.marginals_reinforced[self.connectivity[:,k],0,1]+self.marginals_reinforced[self.connectivity[:,k],1,1])
        self.site_marginals_reinforced=x1/(x0+x1)

        
    def prob_update(self, t, gamma=0.1):
        return torch.rand((self.N))<1-(1+t)**(-gamma)   


    def step_regular_reinforced(self,bias):
        self.psi_old_reinforced=self.psi_reinforced.clone()
        self.psi_reinforced=torch.zeros((self.M,2,2))
        for i in range(2):
            for j in range(2):
                for perm in self.permutations:
                    mult=torch.ones(self.M)
                    for k in range(self.d-1):
                        if perm[k]==1:
                            mult*=self.psi_old_reinforced[self.truncated_connectivity[:,k]][:,perm[k],i]*bias[self.truncated_node_connectivity[:,k]]
                        else:
                            mult*=self.psi_old_reinforced[self.truncated_connectivity[:,k]][:,perm[k],i]*(1-bias[self.truncated_node_connectivity[:,k]])
                            
                    self.psi_reinforced[:,i,j]+=np.exp(self.mu*i)*self.respect_rule(i, j, perm)*mult

        Z=torch.sum(self.psi_reinforced, dim=(1,2))
        self.psi_reinforced[torch.where(Z==0.)]=torch.tensor([[0.25,0.25],[0.25,0.25]])
        Z=torch.where(Z!=0.,Z,1.)
        self.psi_reinforced/=Z.repeat_interleave(4).reshape(self.M,2,2)
        self.psi_reinforced=self.damping_parameter*self.psi_old_reinforced+(1-self.damping_parameter)*self.psi_reinforced
        
            
    def num_mistakes(self, config): 
        num_mistakes=torch.tensor([0])
        outer_density_array=torch.sum(config[self.node_connectivity], dim=(1,))
        for num_alive_neigh in range(self.d+1):
            if not self.respect_rule_v2(0,num_alive_neigh):
                num_mistakes+=torch.sum(torch.logical_and(config==0, outer_density_array==num_alive_neigh))
            if not self.respect_rule_v2(1,num_alive_neigh):
                num_mistakes+=torch.sum(torch.logical_and(config==1, outer_density_array==num_alive_neigh))
                
        return num_mistakes
        
    def reinforced_BP(self,max_iter=None, verbose=False, pi=None, interval_check=1, early_stopping_if_BP_regime=False, num_prints=20, gamma=None):
        bias=torch.rand((self.N,))
        config=torch.round(bias).bool()
        if max_iter is None:
            max_iter=self.max_iter
        if pi is None:
            pi=self.pi
        if gamma is None:
            gamma=self.gamma
            
        if verbose:
            print('=========== Starting BP reinforcement for rule '+str(self.rule)+' with pi = ',pi, ' ===========')
            
        for t in range(max_iter):
            self.step_regular_reinforced(bias)
            self.update_site_marginals()
            sampling=self.prob_update(t, gamma)
            bias=torch.where(torch.logical_and(sampling,self.site_marginals_reinforced<0.5),pi,bias)
            bias=torch.where(torch.logical_and(sampling,self.site_marginals_reinforced>=0.5),1-pi,bias)
            config=torch.round(bias).bool()
            if t%interval_check==0:
                num_mistakes=self.num_mistakes(config)
                diff=self.diff_reinforced()
                if early_stopping_if_BP_regime:
                    if diff<self.tol:
                        if verbose:
                            print('Early stopping since converged to a fixed point but there are still mistakes')
                        return config, False, t
                       
                if verbose and int(max_iter/num_prints)>0 and t%int(max_iter/num_prints)==0:
                    print('# iter: ', t, '  ||    # mistakes: ',num_mistakes.item(),'  ||   BP convergence: ', diff.item())
                elif verbose and int(max_iter/num_prints)==0:
                    print('# iter: ', t, '  ||    # mistakes: ',num_mistakes.item(),'  ||   BP convergence: ', diff.item())                    
                if num_mistakes==0:
                    if verbose:
                        print('Solution found after ', t+1, ' iterations !')
                    self.config=config
                    return config, True, t
        if verbose:
            print('No solution found !')
        self.config=config
        return config, False, t
    
    
    
    def reinforced_BP_search_pi(self,max_iter=None, verbose=1, max_tries=100, interval_check=1, early_stopping_if_BP_regime=True, starting_pi=0.5):
            if max_iter is None:
                max_iter=self.max_iter
            
            best_num_errors= self.N
            best_config=None
            
            
            pi_list=torch.flip(torch.linspace(0,starting_pi,max_tries), dims=(0,))
            for pi in pi_list:
                self.psi_reinforced=torch.rand(size=(self.M,2,2))
                self.psi_reinforced=self.psi_reinforced/self.psi_reinforced.sum(dim=(1,2)).repeat_interleave(4).reshape(self.M,2,2)
                config, found_solution, num_iter=self.reinforced_BP(max_iter=max_iter, verbose=(verbose>=2), pi=pi, interval_check=interval_check, early_stopping_if_BP_regime=early_stopping_if_BP_regime)
                if found_solution:
                    if verbose>=1:
                        print('Solution found for pi = ',pi)
                    return config, True, num_iter, pi
                num_mistakes=self.num_mistakes(config)
                if num_mistakes<best_num_errors:
                    best_config=config
                    best_num_errors=num_mistakes
            self.config=config
            return config, False, num_iter, pi


    def stability_check(self, noise=0.01, p=1., verbose=0):
        # p: probability to add noise to one given message
        psi_copy=self.psi.clone()
        psi_old_copy=self.psi_old.clone()

        # noise on all types of messages (frozen and not frozen)
        # There is a bug in pytorch that torch.normal not automatically located on default device
        self.psi+=torch.abs(torch.normal(0,noise,(self.M,2,2)).to(self.device))*(torch.multinomial(torch.tensor([1.-p,p]), self.M, replacement=True).repeat_interleave(4).reshape(self.M,2,2).double())

        # Normalization
        Z=torch.sum(self.psi, dim=(1,2))
        self.psi[torch.where(Z==0)]=torch.tensor([[0.25,0.25],[0.25,0.25]])
        Z=torch.where(Z!=0.,Z,1.)
        self.psi/=Z.repeat_interleave(4).reshape(self.M,2,2)

        # Run BP
        self.run()

        # Check if close to previous solution or not
        if torch.linalg.norm(self.psi-psi_copy)/self.M<self.tol:
            self.stability=True
        else:
            self.stability=False

        if verbose>=2:
            print('Sample of found fixed points for general stability:\n', self.psi[0], '\n', self.psi[int(self.M/4)],'\n', self.psi[int(self.M/2)])
            print(self)
        if verbose>=1:
            print('Stable : ', self.stability)

        # noise only on non-zero messages:
        self.psi=psi_copy.clone()
        self.psi_old=psi_old_copy.clone()
        self.psi+=torch.abs(torch.normal(0,noise,(self.M,2,2)).to(self.device))*torch.multinomial(torch.tensor([1.-p,p]), self.M, replacement=True).repeat_interleave(4).reshape(self.M,2,2)*(self.psi!=0.).double()
        
        # Normalization
        Z=torch.sum(self.psi, dim=(1,2))
        self.psi[torch.where(Z==0)]=torch.tensor([[0.25,0.25],[0.25,0.25]])
        Z=torch.where(Z!=0.,Z,1.)
        self.psi/=Z.repeat_interleave(4).reshape(self.M,2,2)

        # Run BP
        self.run()
        

        
        # Check if close to previous solution or not
        if torch.linalg.norm(self.psi-psi_copy)/self.M<self.tol:
            self.stability_non_zero=True
        else:
            self.stability_non_zero=False

        if verbose>=2:
            print('Sample of found fixed points for noise only on non-zero components:\n', self.psi[0], '\n', self.psi[int(self.M/4)],'\n', self.psi[int(self.M/2)])
            print(self)
        if verbose>=1:
            print('Stable : ', self.stability_non_zero)

        self.psi=psi_copy
        self.psi_old=psi_old_copy
        self.update_observables()

    def compute_frozen(self, verbose=0):
        self.fraction_hard_fields=(torch.sum(self.psi==1.)/self.M).item()
        self.fraction_i_frozen=(torch.sum(torch.sum(self.psi, dim=(2))==1.)/self.M).item() # i is uniquely determined, i.e. the site marginals is either exactly 0 or 1
        self.fraction_config_forbidden=(torch.sum(torch.sum(self.psi==0., dim=(1,2))>=1)/self.M).item()
        if verbose>=1:
            print('Fraction hard fields : ', self.fraction_hard_fields)
            print('Fraction i frozen: ', self.fraction_i_frozen)
            print('Fraction at least one config forbidden', self.fraction_config_forbidden)

    #=============================================================================================================================
    # Perform multiple search of solutions on multiples graphs
    #=============================================================================================================================    

    # This includes also running BP on the first found solution
    def multiple_runs(self, Ns=[1000,10000,100000], num_runs=5, max_tries=2, save_config=True, verbose=0):
        pi=0.5
        for N in Ns:
            found_solution=False
            num_found=0
            for run in range(num_runs):
                if verbose>0:
                    print('N = '+str(N)+' Try '+str(run+1)+' out of '+str(num_runs))
                self.reset_graph(N=N)
                found=False
                i=0
                while not found and i<max_tries:
                    new_pi=0.5
                    if pi==0.5:
                        config, found, _, new_pi=self.reinforced_BP_search_pi(max_iter=None, verbose=verbose, max_tries=25, interval_check=1, early_stopping_if_BP_regime=True, starting_pi=pi)
                    else:
                        config, found, _, new_pi=self.reinforced_BP_search_pi(max_iter=None, verbose=verbose, max_tries=1, interval_check=1, early_stopping_if_BP_regime=True, starting_pi=pi)
                        if not found:
                            pi=0.5
                            config, found, _, new_pi=self.reinforced_BP_search_pi(max_iter=None, verbose=verbose, max_tries=25, interval_check=1, early_stopping_if_BP_regime=True, starting_pi=pi)
                
                    if found:
                        if not found_solution:
                            found_solution=True
                            self.BP_on_solution()
                            self.stability_check()
                            if save_config:
                                self.save_messages_and_graph()
                        pi=new_pi
                        num_found+=1
                    i+=1
            self.fraction_found=num_found/num_runs
            if num_found<1:
                if verbose>0:
                    print('No solution found for N = '+str(N))
                break
            

    
    #=============================================================================================================================
    # Apply BP on the found configuration
    #=============================================================================================================================                   
            
    def BP_on_solution(self, config=None):
        self.psi=torch.zeros((self.M,2,2))
        if config is None:
            config=self.config
        # Constructing the spiked psi from the solution:
        
        self.psi[:,0,0]=((self.config[self.edges[:,0]]==0)*(self.config[self.edges[:,1]]==0)).double()
        self.psi[:,0,1]=((self.config[self.edges[:,0]]==0)*(self.config[self.edges[:,1]]==1)).double()
        self.psi[:,1,0]=((self.config[self.edges[:,0]]==1)*(self.config[self.edges[:,1]]==0)).double()
        self.psi[:,1,1]=((self.config[self.edges[:,0]]==1)*(self.config[self.edges[:,1]]==1)).double()

        self.update_observables()
        self.psi_old=self.psi.clone()
        self.run()
        self.compute_frozen()

    #=============================================================================================================================
    # Local entropy
    #============================================================================================================================= 
    

    def respect_rule(self, i, j, rest_config):
        outer_density=j+torch.sum(rest_config)
        if self.rule[outer_density]=='0':
            return True if i==0 else False
        elif self.rule[outer_density]=='1':
            return True if i==1 else False
        elif self.rule[outer_density]=='+':
            return True
        elif self.rule[outer_density]=='-':
            return False
    
    def respect_rule_v2(self, i, outer_density):
        if self.rule[outer_density]=='0':
            return True if i==0 else False
        elif self.rule[outer_density]=='1':
            return True if i==1 else False
        elif self.rule[outer_density]=='+':
            return True
        elif self.rule[outer_density]=='-':
            return False
                        
                        
    def diff_LE(self):
        return torch.mean(torch.linalg.norm(self.psi-self.psi_old, dim=(1,2)))
    
    
    def step_LE(self):
        self.psi_old=self.psi.clone()
        self.psi=torch.zeros((self.M,2,2))
        for i in range(2):
            for j in range(2):
                for perm in self.permutations:
                    mult=torch.ones(self.M)
                    for k in range(self.d-1):
                        mult*=self.psi_old[self.truncated_connectivity[:,k]][:,perm[k],i]

                    self.psi[:,i,j]+=np.exp(self.mu*i)*torch.exp(self.kappa*(i!=self.config[self.edges[:,0]]))*self.respect_rule(i, j, perm)*mult

        Z=torch.sum(self.psi, dim=(1,2))
        self.psi[torch.where(Z==0)]=torch.tensor([[0.25,0.25],[0.25,0.25]])
        Z=torch.where(Z!=0.,Z,1.)
        self.psi/=Z.repeat_interleave(4).reshape(self.M,2,2)
        self.psi=self.damping_parameter*self.psi_old+(1-self.damping_parameter)*self.psi
        
 
    def update_observables_LE(self):
        Z_i=torch.zeros(self.N)
        numerator=torch.zeros(self.N)
        numerator_distance=torch.zeros(self.N)
        for i in range(2):
            for j in range(2):
                for perm in self.permutations:
                    mult=self.psi[self.connectivity[:,0]][:,j,i]
                    for k in range(self.d-1):
                        mult*=self.psi[self.connectivity[:,k+1]][:,perm[k],i]
                    Z_i+=np.exp(self.mu*i)*torch.exp(self.kappa*(i!=self.config))*self.respect_rule(i,j,perm)*mult
                    numerator+=i*torch.exp(self.kappa*(i!=self.config))*np.exp(self.mu*i)*self.respect_rule(i,j,perm)*mult
                    numerator_distance+=(i!=self.config)*np.exp(self.mu*i)*torch.exp(self.kappa*(i!=self.config))*self.respect_rule(i,j,perm)*mult
        phi=torch.sum(torch.log(Z_i))-torch.sum(torch.log(torch.sum(self.psi[:self.num_edges]*torch.flip(torch.transpose(self.psi[self.num_edges:],1,2),dims=(0,)),dim=(1,2))))
        rho=torch.sum(numerator/Z_i)
        distance=torch.sum(numerator_distance/Z_i)
        
        self.phi=(phi/self.N).item()
        self.rho=(rho/self.N).item()
        self.distance=(distance/self.N).item()
        
        self.s=self.phi-self.mu*self.rho
        self.local_entropy=self.phi-self.kappa*self.distance
        

    def update_marginals(self):
        self.marginals=torch.zeros((self.M,2,2))
        self.marginals[:,0,0]=self.psi[:,0,0]*torch.flip(self.psi,dims=(0,))[:,0,0]
        self.marginals[:,1,0]=self.psi[:,1,0]*torch.flip(self.psi,dims=(0,))[:,0,1]
        self.marginals[:,0,1]=self.psi[:,0,1]*torch.flip(self.psi,dims=(0,))[:,1,0]
        self.marginals[:,1,1]=self.psi[:,1,1]*torch.flip(self.psi,dims=(0,))[:,1,1]
        Z=torch.sum(self.marginals, dim=(1,2))
        self.marginals[torch.where(Z==0.)]=torch.tensor([[0.25,0.25],[0.25,0.25]])
        Z=torch.where(Z!=0.,Z,1.)
        self.marginals/=Z.repeat_interleave(4).reshape(self.M,2,2)
    
    def run_LE(self,max_iter=None, tol=None, verbose=False):
        if max_iter is None:
            max_iter=self.max_iter
        if tol is None:
            tol=self.tol
        for i in range(max_iter):
            self.step_LE()
            diff=self.diff_LE()
            if verbose>=2 and i%int(max_iter/20)==0:
                print('Iter : ', i+1, 'Diff = ', diff)
            if diff<tol:
                break
        if i==max_iter-1:
            print('No convergence reached for rule ', self.rule, 'after ', i,' steps. Final error: ', self.diff())
        else:
            if verbose:
                print('Converged after ', i, ' steps. Final error: ', self.diff())
        self.update_observables_LE()

    def run_LE_adaptative(self, distance=0, interval_update_kappa=1, adaptative_factor=1, tol_distance=0.001, max_iter=None, tol=None, verbose=False):
        if max_iter is None:
            max_iter=self.max_iter
        if tol is None:
            tol=self.tol
        for i in range(max_iter):
            if i%interval_update_kappa==0:
                self.update_observables_LE()
                self.kappa=self.kappa+(distance-self.distance)*adaptative_factor

            self.step_LE()
            diff=self.diff_LE()
            if verbose>=2 and i%int(max_iter/20)==0:
                print('Iter : ', i+1, 'Diff = ', diff)
                print('Distance difference : ', np.abs(distance-self.distance), ' kappa : ', self.kappa)
            
            if diff<tol and np.abs(distance-self.distance)<tol_distance:
                break
        if i==max_iter-1:
            print('No convergence reached for rule ', self.rule, 'after ', i,' steps. Final error: ', self.diff())
            print('Distance difference : ', np.abs(distance-self.distance), ' kappa : ', self.kappa)

        else:
            if verbose:
                print('Converged after ', i, ' steps. Final error: ', self.diff())
                print('Distance difference : ', np.abs(distance-self.distance), ' kappa : ', self.kappa)

        self.update_observables_LE()
        
            

    #=============================================================================================================================
    # Saving solutions
    #=============================================================================================================================
    
    def save_parameters(self, path=None):
        if path is None:
            path='results/'+str(self.rule)+'/BP_reinf'
            if not os.path.exists(path):
                os.makedirs(path)
            path+='/parameters.json'
        parameters={'rule': self.rule, 'N': self.N, 'chemical potential': self.mu, 'tolerance': self.tol, 'maximum number of iterations': self.max_iter, 'damping parameter': self.damping_parameter, 'bias pi':self.pi, 'bias gamma':self.gamma}
        with open(path, 'w+') as json_file:
            json.dump(parameters, json_file) 


    def save_observables(self, path=None):
        if path is None:
            path='results/'+str(self.rule)+'/BP_reinf'
            if not os.path.exists(path):
                os.makedirs(path)
            path+='/observables.json'
        observables={'Free entropy density': self.phi, 'density': self.rho, 'entropy density': self.s, 'stability': self.stability, 'stability non zero': self.stability_non_zero, 'fraction hard fields': self.fraction_hard_fields, 'fraction i frozen': self.fraction_i_frozen, 'fraction config forbidden': self.fraction_config_forbidden, 'fraction found': self.fraction_found}
        with open(path, 'w+') as json_file:
            json.dump(observables, json_file)

    def save_messages_and_graph(self, path=None):
        if path is None:
            path='results/'+str(self.rule)+'/BP_reinf'
            if not os.path.exists(path):
                os.makedirs(path)
        path_messages=path+'/messages.npy'
        path_messages_reinf=path+'/messages_reinf.npy'
        path_edges=path+'/edges.npy'
        path_config=path+'/config.npy'
        np.save(path_messages, self.psi.detach().cpu().numpy())
        np.save(path_messages_reinf, self.psi_reinforced.detach().cpu().numpy())
        np.save(path_edges, self.edges.detach().cpu().numpy())
        np.save(path_config, self.config.detach().cpu().numpy())

    def save(self, folder=None, save_messages_and_graph=False, parameters_file='parameters.json', observables_file='observables.json'):
        if folder is None:
            folder='results/'+str(self.rule)+'/BP_reinf'
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.save_parameters(path=folder+'/'+parameters_file)
        self.save_observables(path=folder+'/'+observables_file)
        if save_messages_and_graph:
            self.save_population(path=folder)


    def load_parameters(self, path=None):
        if path is None:
            path='results/'+str(self.rule)+'/BP_reinf='+'/parameters.json'
        if os.path.exists(path):
            parameters=json.load(open(path))
            self.rule=parameters['rule']
            self.N=parameters['N']
            self.mu=parameters['chemical potential']
            self.tol=parameters['tolerance']
            self.max_iter=parameters['maximum number of iterations']
            self.damping_parameter=parameters['damping parameter']
            self.pi=parameters['bias pi']
            self.gamma=parameters['bias gamma']
            
            self.d=len(self.rule)-1
            self.permutations=np.array(list(itertools.product([0,1], repeat=self.d-1)))
            self.permutations=torch.tensor(self.permutations)

        else:
            print('Warning ! The parameters were not loaded. The file '+path+' does not exist !')
            
    def load_observables(self, path=None):
        if path is None:
            path='results/'+str(self.rule)+'/BP_reinf='+'/observables.json'
        if os.path.exists(path):
            observables=json.load(open(path))
            self.phi=observables['Free entropy density']
            self.rho=observables['density']
            self.s=observables['entropy density']
            self.stability=observables['stability']
            self.stability_non_zero=observables['stability non zero']
            self.fraction_hard_fields=observables['fraction hard fields']
            self.fraction_i_frozen=observables['fraction i frozen']
            self.fraction_config_forbidden=observables['fraction config forbidden']
            self.fraction_found=observables['fraction found']
                
        else:
            print('Warning ! The observables were not loaded. The file '+path+' does not exist !')

    def load_messages_and_graph(self, path=None):
        if path is None:
            path='results/'+str(self.rule)+'/BP_reinf'
        path_messages=path+'/messages.npy'
        path_messages_reinf=path+'/messages_reinf.npy'
        path_edges=path+'/edges.npy'
        path_config=path+'/config.npy'
        
        self.psi=torch.tensor(np.load(path_messages)).to(self.device) if os.path.exists(path_messages) else print('Warning ! The messages was not loaded. The file '+path_messages+' does not exist !')
        self.psi_reinforced=torch.tensor(np.load(path_messages_reinf)).to(self.device) if os.path.exists(path_messages_reinf) else print('Warning ! The population was not loaded. The file '+path_messages_reinf+' does not exist !')    
        self.edges=torch.tensor(np.load(path_edges)).to(self.device) if os.path.exists(path_edges) else print('Warning ! The graph was not loaded. The file '+path_edges+' does not exist !')    
        self.reset_graph(self.edges)
        self.config=torch.tensor(np.load(path_config)).to(self.device) if os.path.exists(path_config) else print('Warning ! The configuration was not loaded. The file '+path_config+' does not exist !') 

        self.psi_old=self.psi.clone()
        self.psi_old_reinforced=self.psi_reinforced.clone()

    def load(self, load_messages_and_graph=False, folder=None, parameters_file='parameters.json', observables_file='observables.json'):
        if folder is None:
            folder='results/'+str(self.rule)+'/BP_reinf'
        self.load_parameters(path=folder+'/'+parameters_file)
        self.load_observables(path=folder+'/'+observables_file)
        if load_messages_and_graph:
            self.load_messages_and_graph(path=folder)
        

        
