import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy import optimize

import os
import json
import csv
import ast

        
#===============================================================================================

def fitting_func(x,a,b,c,d):
    return a+b*np.power(2,x)+c*np.power(3,x)



class population_dynamics:
    def __init__(self, rule, m=1, mu=0, M=10000, num_samples=100000, damping_parameter=0.8, hard_fields=True, planted_messages=None, fraction_planted_messages=1, noise=0, impose_symmetry=False, max_iter=10000, tol=10, convergence_check_interval=200, sampling_threshold=8000, sampling_interval=50, m_list=np.linspace(0.00001, 1, 30), seed=None):
        self.rule=rule
        self.m=m
        self.mu=mu
        self.M=M
        self.num_samples=num_samples
        self.damping_parameter=0.8
        self.hard_fields=hard_fields
        self.planted_messages=planted_messages
        self.noise=noise
        self.fraction_planted_messages=fraction_planted_messages
        self.impose_symmetry=impose_symmetry
        self.max_iter=max_iter
        self.tol=tol
        self.convergence_check_interval=convergence_check_interval
        self.sampling_threshold=sampling_threshold
        self.sampling_interval=sampling_interval
        self.m_list=m_list
        self.seed=seed
        
        self.rng=np.random.default_rng()
        if seed is not None:
            self.rng=np.random.default_rng(seed)
            

        self.d=len(rule)-1
        self.d_min_1=self.d-1
        self.permutations=np.array(list(itertools.product([0,1], repeat=self.d_min_1)))
        self.nb_new_samples=round((1-self.damping_parameter)*self.M)   
        self.d_min_1_times_num_samples=self.nb_new_samples*self.d_min_1


        
        self.population=np.zeros((M,2,2))
        if self.hard_fields:
            self.population=self.rng.choice(np.array([[[1.,0],[0,0]], [[0.,1],[0,0]], [[0.,0],[1,0]], [[0.,0],[0,1]]]), self.M)
            if self.planted_messages is not None:
                self.population[:round(self.M*self.fraction_planted_messages)]=np.maximum((self.planted_messages[:round(self.M*self.fraction_planted_messages)]+self.rng.normal(0, self.noise,(round(self.M*self.fraction_planted_messages),2,2))),0)
                Z=self.population[:round(self.M*self.fraction_planted_messages)].sum(axis=(1,2))
                self.population[np.where(Z==0)]=np.array([[0.25,0.25],[0.25,0.25]])
                Z=np.where(Z==0, 1, Z)
                self.population[:round(self.M*self.fraction_planted_messages)]=self.population[:round(self.M*self.fraction_planted_messages)]/np.repeat(Z, repeats=4).reshape(round(self.M*self.fraction_planted_messages),2,2)
        else:
            self.population=self.rng.uniform(size=(M,2,2))
            if self.planted_messages is not None:
                self.population[:round(self.M*self.fraction_planted_messages)]=np.maximum((self.planted_messages[:round(self.M*self.fraction_planted_messages)]+self.rng.normal(0, self.noise,(round(self.M*self.fraction_planted_messages),2,2))),0)
            Z=self.population.sum(axis=(1,2))
            self.population[np.where(Z==0)]=np.array([[0.25,0.25],[0.25,0.25]])
            Z=np.where(Z==0, 1, Z)
            self.population=self.population/np.repeat(Z, repeats=4).reshape(self.M,2,2)
                
        self.old_population=self.population.copy()
        
        self.no_update=False
        
        self.psi=None
        self.phi=None
        self.complexity=None
        self.rho=None
        self.s=None

        self.psi_mean=None
        self.phi_mean=None
        self.complexity_mean=None
        self.rho_mean=None
        self.s_mean=None
        
 
        self.psi_std=None
        self.phi_std=None
        self.complexity_std=None
        self.rho_std=None
        self.s_std=None
        
        self.phi_list=None
        self.phi_list_std=None
        self.complexity_list=None
        self.complexity_list_std=None
        self.psi_list=None
        self.psi_list_std=None
        self.rho_list=None
        self.rho_list_std=None
        self.s_list=None
        self.s_list_std=None
        
        self.fitting_param_phi=None
        self.fitting_param_rho=None
        
        self.rho_s=None
        self.rho_d=None
        self.phi_s=None
        self.phi_d=None
                    

        
    def __repr__(self):
        description="Instance of class \'population_dynamics\'\nRule : "+str(self.rule)+"\nμ =  "+str(self.mu)+"\nPopulation size: "+str(self.M)
        if self.phi_mean is not None and self.rho_s is None:
            description+='\nφ = '+str(self.phi_mean)+' +/- '+str(self.phi_std)
            description+='\nΣ = '+str(self.complexity_mean)+' +/- '+str(self.complexity_std)
            description+='\nΨ = '+str(self.psi_mean)+' +/- '+str(self.psi_std)
            description+='\nρ = '+str(self.rho_mean)+' +/- '+str(self.rho_std)
            description+='\ns = '+str(self.s_mean)+' +/- '+str(self.s_std)
        if self.rho_s is not None:
            description+='\nφ_s = '+str(self.phi_s)
            description+='\nρ_s = '+str(self.rho_s)
        return description
    
    #=============================================================================================================================
    # Running the population dynamics
    #=============================================================================================================================
        
    def respect_rule(self,i, j, rest_config):
        outer_density=j+np.sum(rest_config)
        if self.rule[outer_density]=='0':
            return True if i==0 else False
        elif self.rule[outer_density]=='1':
            return True if i==1 else False
        elif self.rule[outer_density]=='+':
            return True
        elif self.rule[outer_density]=='-':
            return False
    
    
    def step(self):
        if self.no_update==False:
            self.old_population=self.population.copy()
        psi=self.population[self.rng.choice(self.M, self.d_min_1_times_num_samples)].reshape(self.nb_new_samples,self.d_min_1,2,2)
        psi_new=np.zeros((self.nb_new_samples,2,2))
        for i in range(2):
            for j in range(2):
                for perm in self.permutations:
                    mult=np.ones(self.nb_new_samples)
                    for l, k in enumerate(perm):
                        mult*=psi[:,l,k,i]
                    psi_new[:,i,j]+=np.exp(self.mu*i)*self.respect_rule(i, j, perm)*mult
        Z=np.sum(psi_new, axis=(1,2))
        sum_Z=np.sum(Z)
        if sum_Z!=0:
            indexes=self.rng.choice(self.nb_new_samples, self.nb_new_samples, p=np.power(Z, self.m)/np.sum(np.power(Z, self.m)))
            if self.impose_symmetry:
                mid_index=round(self.nb_new_samples/2)
                psi_new=np.concatenate(np.flip(psi_new[indexes[:mid_index]],psi_new[indexes[mid_index:]], axis=(1,2)))
            else:
                psi_new=psi_new[indexes]
            Z=Z[indexes]
            psi_new/=np.repeat(Z, repeats=4).reshape(self.nb_new_samples,2,2)
            self.population[self.rng.choice(self.M, self.nb_new_samples)]=psi_new
            self.no_update=False
        else:
            self.no_update=True
        
    
    def run(self, max_iter=None, tol=None, check_convergence=True, convergence_check_interval=None, sampling_threshold=None, sampling_interval=None, reset_population=False, verbose=0):
        if reset_population:
            self.reset_population()
        if max_iter is None:
            max_iter=self.max_iter
        if tol is None:
            tol=self.tol
        if convergence_check_interval is None:
            convergence_check_interval=self.convergence_check_interval
        if sampling_threshold is None:
            sampling_threshold=self.sampling_threshold
        if sampling_interval is None:
            sampling_interval=self.sampling_interval
        
        phi_list=[]
        complexity_list=[]
        psi_list=[]
        rho_list=[]
        s_list=[]
        for i in range(max_iter):
            self.step()
            if check_convergence and i%convergence_check_interval==0:
                diff=self.diff()
                if verbose>=2:
                    print('Difference after ', i+1, 'iterations : ', diff)
                if diff<tol:
                    if verbose>=1:
                        print('Early stopping ! Convergence of ', tol, 'reached after ', i+1, 'iterations.')
                    break
            if i%sampling_interval==0 and i>=sampling_threshold:
                self.update_observables()
                phi_list.append(self.phi)
                complexity_list.append(self.complexity)
                psi_list.append(self.psi)
                rho_list.append(self.rho)
                s_list.append(self.s)
        self.step()
        self.update_observables()
        phi_list.append(self.phi)
        complexity_list.append(self.complexity)
        psi_list.append(self.psi)
        rho_list.append(self.rho)
        s_list.append(self.s)
        if verbose>=1:
            print("Finished ! Final difference: ", self.diff())
        self.phi_mean, self.phi_std=np.mean(phi_list), np.std(phi_list)
        self.complexity_mean, self.complexity_std=np.mean(complexity_list), np.std(complexity_list)
        self.psi_mean, self.psi_std=np.mean(psi_list), np.std(psi_list)
        self.rho_mean, self.rho_std=np.mean(rho_list), np.std(rho_list)
        self.s_mean, self.s_std=np.mean(s_list), np.std(s_list)
        
    
    def update_observables(self, num_samples=None):
        if num_samples==None:
            num_samples=self.num_samples

        pop_sample_Z_i=self.population[self.rng.integers(0,self.M,size=num_samples*self.d)].reshape((num_samples,self.d,2,2))
        pop_sample_Z_ij=self.population[self.rng.integers(0,self.M,size=num_samples*2)].reshape((num_samples,2,2,2))
        
        _i=np.zeros(num_samples)
        _ij=np.zeros(num_samples)
        _i_prime=np.zeros(num_samples)
        for i in range(2):
            for j in range(2):
                for perm in self.permutations:
                    mult=pop_sample_Z_i[:,0,j,i].copy()
                    for l, k in enumerate(perm):
                        mult*=pop_sample_Z_i[:,l+1,k,i]
                    _i+=np.exp(self.mu*i)*self.respect_rule(i, j, perm)*mult
                    _i_prime+=i*np.exp(self.mu*i)*self.respect_rule(i, j, perm)*mult
                    
                _ij+=pop_sample_Z_ij[:,0,i,j]*pop_sample_Z_ij[:,1,j,i]
                
        power=np.power(_i,self.m)
        Z_i=np.sum(power)
        Z_i_deriv=np.sum(power*np.where(_i!=0.,np.log(_i),0.))

        power=np.power(_ij,self.m)
        Z_ij=np.sum(power)
        Z_ij_deriv=np.sum(power*np.where(_ij!=0,np.log(_ij),0))

        Z_i=Z_i/num_samples
        Z_i_deriv=Z_i_deriv/num_samples

        Z_ij=Z_ij/num_samples
        Z_ij_deriv=Z_ij_deriv/num_samples

        self.psi=np.log(Z_i)-self.d/2*np.log(Z_ij)
        self.phi=Z_i_deriv/Z_i-self.d/2*Z_ij_deriv/Z_ij
        self.complexity=self.psi-self.m*self.phi
        self.rho=1/Z_i*np.sum(_i_prime*np.power(_i,self.m-1))/num_samples
        self.s=self.phi-self.mu*self.rho
        
    def reset_population(self):
        self.population=np.zeros((self.M,2,2))
        if self.hard_fields:
            self.population=self.rng.choice(np.array([[[1.,0],[0,0]], [[0.,1],[0,0]], [[0.,0],[1,0]], [[0.,0],[0,1]]]), self.M)
            if self.planted_messages is not None:
                self.population[:round(self.M*self.fraction_planted_messages)]=self.planted_messages[:round(self.M*self.fraction_planted_messages)] 
        else:
            self.population=self.rng.uniform(size=(M,2,2))
            if self.planted_messages is not None:
                self.population[:round(self.M*self.fraction_planted_messages)]=self.plantent_messages[:round(self.M*self.fraction_planted_messages)]
                
        
        
    def diff(self, n_bins=1000):    
        old_hist_00=np.histogram(self.old_population[:,0,0], bins=n_bins, range=(0,1), density=True)
        old_hist_01=np.histogram(self.old_population[:,0,1], bins=n_bins, range=(0,1), density=True)
        old_hist_10=np.histogram(self.old_population[:,1,0], bins=n_bins, range=(0,1), density=True)
        old_hist_11=np.histogram(self.old_population[:,1,1], bins=n_bins, range=(0,1), density=True)
        new_hist_00=np.histogram(self.population[:,0,0], bins=n_bins, range=(0,1), density=True)
        new_hist_01=np.histogram(self.population[:,0,1], bins=n_bins, range=(0,1), density=True)
        new_hist_10=np.histogram(self.population[:,1,0], bins=n_bins, range=(0,1), density=True)
        new_hist_11=np.histogram(self.population[:,1,1], bins=n_bins, range=(0,1), density=True)
        return np.sum(np.abs(new_hist_00[0]-old_hist_00[0])+np.abs(new_hist_01[0]-old_hist_01[0])+np.abs(new_hist_10[0]-old_hist_10[0])+np.abs(new_hist_11[0]-old_hist_11[0]))
        

    def draw_population(self,n_bins=100, title=None, old=False):
        f, ax=plt.subplots(2,2)
        f.suptitle(title)
        xlabels=[[r'$\psi_{0,0}$',r'$\psi_{0,1}$'],[r'$\psi_{1,0}$',r'$\psi_{1,1}$']]
        for i in range(2):
            for j in range(2):
                if not old:
                    pop=self.population[:,i,j]
                else:
                    pop=self.old_population[:,i,j]
                ax[i,j].hist(pop, bins=n_bins, range=(0,1), density=True)
                ax[i,j].set_xlabel(xlabels[i][j])
                ax[i,j].set_ylabel('Approximated distribution')
                ax[i,j].set_xlim((0,1))
            
        plt.tight_layout()
              
    #=============================================================================================================================
    # complexity curves
    #=============================================================================================================================                
 
    def compute_complexity_curves(self, m_list=None, check_convergence=False, verbose=0):
        if m_list is not None:
            self.m_list=m_list
        
        num_m=len(self.m_list)
        self.phi_list=np.zeros(num_m)
        self.phi_list_std=np.zeros(num_m)
        self.complexity_list=np.zeros(num_m)
        self.complexity_list_std=np.zeros(num_m)
        self.psi_list=np.zeros(num_m)
        self.psi_list_std=np.zeros(num_m)
        self.rho_list=np.zeros(num_m)
        self.rho_list_std=np.zeros(num_m)
        self.s_list=np.zeros(num_m)
        self.s_list_std=np.zeros(num_m)
        
        for i, m in enumerate(self.m_list):
            if verbose>=1:
                print('========== m = ', m, ' ==========')
            self.m=m
            self.run(check_convergence=check_convergence, reset_population=True, verbose=verbose)
            self.phi_list[i]=self.phi_mean
            self.complexity_list[i]=self.complexity_mean
            self.psi_list[i]=self.psi_mean
            self.rho_list[i]=self.rho_mean
            self.s_list[i]=self.s_mean
            self.phi_list_std[i]=self.phi_std
            self.complexity_list_std[i]=self.complexity_std
            self.rho_list_std[i]=self.rho_std
            self.s_list_std[i]=self.s_std
            
        if np.max(self.complexity_list)<0:
            if verbose>=1:
                print("No intersection with the complexity=0 line !")
            self.phi_s=np.max(self.phi_list)
            self.rho_s=np.max(self.rho_list)    
        else:                
            index_max=np.argmax(self.complexity_list)

            self.fitting_param_phi, _ = optimize.curve_fit(fitting_func, self.phi_list[index_max:], self.complexity_list[index_max:], method="lm")
            phi_samples=np.linspace(min(self.phi_list),max(self.phi_list),100000)
            self.phi_s=phi_samples[np.argmin(np.abs(fitting_func(phi_samples, *self.fitting_param_phi)))]
            self.phi_d=self.phi_list[index_max]

            self.fitting_param_rho, _ = optimize.curve_fit(fitting_func, self.rho_list[index_max:], self.complexity_list[index_max:], method="lm")
            rho_samples=np.linspace(min(self.rho_list),max(self.rho_list),100000)
            self.rho_s=rho_samples[np.argmin(np.abs(fitting_func(rho_samples, *self.fitting_param_rho)))]
            self.rho_d=self.rho_list[index_max]

        
    def draw_sigma_phi(self, errorbars=False, title=None):
        if title is None:
            title="$\mu = $"+str(self.mu)
            
        f, ax=plt.subplots()
        ax.set_title(title)
        
        ax.axhline(0, linestyle='--', color='k', markersize=1)
        ax.axvline(self.phi_s, linestyle='--', color='grey', markersize=1)

        samples=np.linspace(min(self.phi_list),max(self.phi_list),1000)
        ax.plot(samples, fitting_func(samples, *self.fitting_param_phi))
        ax.plot(self.phi_list, self.complexity_list, 'x')
        
        if errorbars:
            ax.errorbar(self.phi_list, self.complexity_list, xerr=self.phi_list_std, yerr=self.complexity_list_std, fmt='.', capsize=3)
            
        ax.set_xlabel(r'$\phi$')
        ax.set_ylabel(r'$\Sigma$');
            
    def draw_sigma_rho(self, errorbars=False, title=None):
        if title is None:
            title="$\mu = $"+str(self.mu)
            
        f, ax=plt.subplots()
        ax.set_title(title)
        
        ax.axhline(0, linestyle='--', color='k', markersize=1)
        ax.axvline(self.rho_s, linestyle='--', color='grey', markersize=1)

        samples=np.linspace(min(self.rho_list),max(self.rho_list),1000)
        ax.plot(samples, fitting_func(samples, *self.fitting_param_rho))
        ax.plot(self.rho_list, self.complexity_list, 'x')
        
        if errorbars:
            ax.errorbar(self.rho_list, self.complexity_list, xerr=self.rho_list_std, yerr=self.complexity_list_std, fmt='.', capsize=3)
            
        ax.set_xlabel(r'$\rho$')
        ax.set_ylabel(r'$\Sigma$');
        
    #=============================================================================================================================
    # Saving and loading
    #=============================================================================================================================
    def save_parameters(self, path=None):
        if path is None:
            path='results/'+str(self.rule)+'/population_dynamics_mu='+str(self.mu)
            if not os.path.exists(path):
                os.makedirs(path)
            path+='/parameters.json'
        parameters={'rule': self.rule, 'Parisi parameter': self.m, 'chemical potential': self.mu, 'population size': self.M, 'number of samples': self.num_samples, 'hard fields': self.hard_fields, 'planted messages': self.planted_messages, 'fraction planted messages': self.fraction_planted_messages, 'impose symmetry': self.impose_symmetry , 'maximum number of iterations': self.max_iter, 'damping parameter': self.damping_parameter, 'tolerance': self.tol, 'convergence check interval': self.convergence_check_interval, 'sampling threshold': self.sampling_threshold, 'sampling interval': self.sampling_interval, 'Parisi parameter list': self.m_list.tolist(), 'seed': self.seed}
        with open(path, 'w+') as json_file:
            json.dump(parameters, json_file)
            
    def save_observables(self, path=None):
        if path is None:
            path='results/'+str(self.rule)+'/population_dynamics_mu='+str(self.mu)
            if not os.path.exists(path):
                os.makedirs(path)
            path+='/observables.json'
        observables={'replicated free entropy': self.psi, 'cluster free entropy': self.phi, 'complexity': self.complexity, 'density': self.rho, 'entropy': self.s, 'replicated free entropy mean': self.psi_mean, 'replicated free entropy std': self.psi_std, 'cluster free entropy mean': self.phi_mean, 'cluster free entropy std': self.phi_std, 'complexity mean': self.complexity_mean, 'complexity std': self.complexity_std, 'density mean': self.rho_mean, 'density std': self.rho_std, 'entropy mean': self.s_mean, 'entropy std': self.s_std}
        with open(path, 'w+') as json_file:
            json.dump(observables, json_file)
            
    def save_complexity_curves(self, path_lists=None, path_parameters=None):

        if self.phi_list is not None:
            if path_lists is None:
                path_lists='results/'+str(self.rule)+'/population_dynamics_mu='+str(self.mu)
                if not os.path.exists(path_lists):
                    os.makedirs(path_lists)
                path_lists+='/complexity_curves.csv'


            with open(path_lists, 'w', newline='') as csv_file:
                    writer=csv.writer(csv_file)
                    writer.writerow(['replicated free entropy mean', 'replicated free entropy std', 'cluster free entropy mean', 'cluster free entropy std', 'complexity mean', 'complexity std', 'density mean', 'density std', 'entropy mean', 'entropy std'])
                    for i in range(len(self.phi_list)):
                        writer.writerow([self.psi_list[i], self.psi_list_std[i], self.phi_list[i], self.phi_list_std[i], self.complexity_list[i], self.complexity_list_std[i], self.rho_list[i], self.rho_list_std[i], self.s_list[i], self.s_list_std[i]])
                        
            if path_parameters is None:
                path_parameters='results/'+str(self.rule)+'/population_dynamics_mu='+str(self.mu)
                if not os.path.exists(path_parameters):
                    os.makedirs(path_parameters)
                path_parameters+='/complexity_curves.json'
            parameters=None
            if self.fitting_param_phi is None:
                parameters={'fitting parameters cluster free entropy': None, 'fitting parameters density': None, 'dynamical transition cluster free entropy': self.phi_d, 'static transition cluster free entropy': self.phi_s, 'dynamical transition density': self.rho_d, 'static transition density': self.rho_s}
            else:
                parameters={'fitting parameters cluster free entropy': self.fitting_param_phi.tolist(), 'fitting parameters density': self.fitting_param_rho.tolist(), 'dynamical transition cluster free entropy': self.phi_d, 'static transition cluster free entropy': self.phi_s, 'dynamical transition density': self.rho_d, 'static transition density': self.rho_s}
                
            with open(path_parameters, 'w+') as json_file:
                json.dump(parameters, json_file)
                
                
    def save_population(self, path=None):
        if path is None:
            path='results/'+str(self.rule)+'/population_dynamics_mu='+str(self.mu)
            if not os.path.exists(path):
                os.makedirs(path)
            path+='/population.npy'
        np.save(path, self.population)
        
        
    def save(self, folder=None, save_population=False, parameters_file='parameters.json', observables_file='observables.json', 
             complexity_curves_lists='complexity_curves.csv',
             complexity_curves_parameters='complexity_curves.json', population_file='population.npy'):
        if folder is None:
            folder='results/'+str(self.rule)+'/population_dynamics_mu='+str(self.mu)
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.save_parameters(path=folder+'/'+parameters_file)
        self.save_observables(path=folder+'/'+observables_file)
        self.save_complexity_curves(path_lists=folder+'/'+complexity_curves_lists, path_parameters=folder+'/'+complexity_curves_parameters)
        if save_population:
            self.save_population(path=folder+'/'+population_file)
                                    
                 
            
    def load_parameters(self, path=None):
        if path is None:
            path='results/'+str(self.rule)+'/population_dynamics_mu='+str(self.mu)+'/parameters.json'
        if os.path.exists(path):
            parameters=json.load(open(path))
            self.rule=parameters['rule']
            self.m=parameters['Parisi parameter']
            self.mu=parameters['chemical potential']
            self.M=parameters['population size']
            self.num_samples=parameters['number of samples']
            self.hard_fields=parameters['hard fields']
            self.planted_messages=parameters['planted messages']
            self.fraction_planted_messages=parameters['fraction planted messages']
            self.impose_symmetry=parameters['impose symmetry']
            self.max_iter=parameters['maximum number of iterations']
            self.damping_parameter=parameters['damping parameter']
            self.tol=parameters['tolerance']
            self.convergence_check_interval=parameters['convergence check interval']
            self.sampling_threshold=parameters['sampling threshold']
            self.sampling_interval=parameters['sampling interval']
            self.m_list=np.array(parameters['Parisi parameter list'])
            self.seed=parameters['seed']
            
            self.rng=np.random.default_rng()
            if self.seed is not None:
                self.rng=np.random.default_rng(self.seed)

            self.d=len(self.rule)-1
            self.d_min_1=self.d-1
            self.permutations=np.array(list(itertools.product([0,1], repeat=self.d_min_1)))
            self.nb_new_samples=round((1-self.damping_parameter)*self.M)   
            self.d_min_1_times_num_samples=self.nb_new_samples*self.d_min_1        

        else:
            print('Warning ! The parameters were not loaded. The file '+path+' does not exist !')   
            
    def load_observables(self, path=None):
        if path is None:
            path='results/'+str(self.rule)+'/population_dynamics_mu='+str(self.mu)+'/observables.json'
        if os.path.exists(path):
            observables=json.load(open(path))
            self.psi=observables['replicated free entropy']
            self.phi=observables['cluster free entropy']
            self.complexity=observables['complexity']
            self.rho=observables['density']
            self.s=observables['entropy']

            self.phi_mean=observables['cluster free entropy mean']
            self.complexity_mean=observables['complexity mean']
            self.psi_mean=observables['replicated free entropy mean']
            self.rho_mean=observables['density mean']
            self.s_mean=observables['entropy mean']

            self.phi_std=observables['cluster free entropy std']
            self.complexity_std=observables['complexity std']
            self.psi_std=observables['replicated free entropy std']
            self.rho_std=observables['density std']
            self.s_std=observables['entropy std']
        else:
            print('Warning ! The observables were not loaded. The file '+path+' does not exist !')

    def load_complexity_curves(self, path_lists=None, path_parameters=None):
        if path_lists is None:
            path_lists='results/'+str(self.rule)+'/population_dynamics_mu='+str(self.mu)+'complexity_curves.csv'
        if os.path.exists(path_lists):
            csv_reader=csv.reader(open(path_lists))
            self.psi_list=[]
            self.psi_list_std=[]
            self.phi_list=[]
            self.phi_list_std=[]
            self.complexity_list=[]
            self.complexity_list_std=[]
            self.rho_list=[]
            self.rho_list_std=[]
            self.s_list=[]
            self.s_list_std=[]
            for i, row in enumerate(csv_reader):
                if i!=0:
                    self.psi_list.append(ast.literal_eval(row[0]))
                    self.psi_list_std.append(ast.literal_eval(row[1]))
                    self.phi_list.append(ast.literal_eval(row[2]))
                    self.phi_list_std.append(ast.literal_eval(row[3]))
                    self.complexity_list.append(ast.literal_eval(row[4]))
                    self.complexity_list_std.append(ast.literal_eval(row[5]))
                    self.rho_list.append(ast.literal_eval(row[6]))
                    self.rho_list_std.append(ast.literal_eval(row[7]))
                    self.s_list.append(ast.literal_eval(row[8]))
                    self.s_list_std.append(ast.literal_eval(row[9]))
            self.psi_list=np.array(self.psi_list)
            self.psi_list_std=np.array(self.psi_list_std)
            self.phi_list=np.array(self.phi_list)
            self.phi_list_std=np.array(self.phi_list_std)
            self.complexity_list=np.array(self.complexity_list)
            self.complexity_list_std=np.array(self.complexity_list_std)
            self.rho_list=np.array(self.rho_list)
            self.rho_list_std=np.array(self.rho_list_std)
            self.s_list=np.array(self.s_list)
            self.s_list_std=np.array(self.s_list_std)
        if path_parameters is None:
            path_parameters='results/'+str(self.rule)+'/population_dynamics_mu='+str(self.mu)+'complexity_curves.json'
        if os.path.exists(path_parameters):
            parameters=json.load(open(path_parameters))
            self.fitting_param_phi=np.array(parameters['fitting parameters cluster free entropy']) if parameters['fitting parameters cluster free entropy'] is not None else None
            self.fitting_param_rho=np.array(parameters['fitting parameters density']) if parameters['fitting parameters density'] is not None else None
            self.phi_d=parameters['dynamical transition cluster free entropy']
            self.phi_s=parameters['static transition cluster free entropy']
            self.rho_d=parameters['dynamical transition density']
            self.rho_s=parameters['static transition density']
            
            
    def load_population(self, path=None):
        if path is None:
            path='results/'+str(self.rule)+'/population_dynamics_mu='+str(self.mu)+'/population.npy'
        if os.path.exists(path):
            self.population=np.load(path)
        else:
            print('Warning ! The population was not loaded. The file '+path+' does not exist !')
            

                                                                    
            
    def load(self, load_population=False, folder=None, parameters_file='parameters.json', observables_file='observables.json', complexity_curves_lists='complexity_curves.csv', complexity_curves_parameters='complexity_curves.json', population_file='population.npy'):
        if folder is None:
            folder='results/'+str(self.rule)+'/population_dynamics_mu='+str(self.mu)+'/'
        self.load_parameters(path=folder+'/'+parameters_file)
        self.load_observables(path=folder+'/'+observables_file)
        self.load_complexity_curves(path_lists=folder+'/'+complexity_curves_lists, path_parameters=folder+'/'+complexity_curves_parameters)
        if load_population:
            self.load_population(path=folder+'/'+population_file)
        


           
        
                

        

        
        
