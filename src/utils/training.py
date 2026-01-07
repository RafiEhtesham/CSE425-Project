"""
Training utilities for VAE models
"""

import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np


def train_vae(model, train_loader, val_loader, num_epochs, learning_rate, device, beta=1.0, verbose=True):
    """
    Train a VAE model
    
    Args:
        model: VAE model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        beta: Beta parameter for KL divergence
        verbose: Print progress
    
    Returns:
        model, history
    """
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from models.vae import vae_loss
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_recon_loss': [],
        'val_recon_loss': [],
        'train_kld': [],
        'val_kld': []
    }
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_recon_loss = 0
        train_kld = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', disable=not verbose)
        for batch in pbar:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x, _ = batch
            else:
                x = batch
            
            x = x.to(device) #type: ignore
            
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            loss, recon_loss, kld = vae_loss(recon_x, x, mu, logvar, beta)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_kld += kld.item()
            
            pbar.set_postfix({'loss': loss.item() / len(x)})
        
        train_loss /= len(train_loader.dataset)
        train_recon_loss /= len(train_loader.dataset)
        train_kld /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0
        val_recon_loss = 0
        val_kld = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    x, _ = batch
                else:
                    x = batch
                
                x = x.to(device) #type: ignore
                recon_x, mu, logvar = model(x)
                loss, recon_loss, kld = vae_loss(recon_x, x, mu, logvar, beta)
                
                val_loss += loss.item()
                val_recon_loss += recon_loss.item()
                val_kld += kld.item()
        
        val_loss /= len(val_loader.dataset)
        val_recon_loss /= len(val_loader.dataset)
        val_kld /= len(val_loader.dataset)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_recon_loss'].append(train_recon_loss)
        history['val_recon_loss'].append(val_recon_loss)
        history['train_kld'].append(train_kld)
        history['val_kld'].append(val_kld)
        
        if verbose:
            print(f'Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')
    
    return model, history


def train_cvae(model, train_loader, val_loader, num_epochs, learning_rate, device, num_classes, beta=1.0, verbose=True):
    """
    Train a Conditional VAE model
    """
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from models.vae import cvae_loss
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_recon_loss': [],
        'val_recon_loss': [],
        'train_kld': [],
        'val_kld': []
    }
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_recon_loss = 0
        train_kld = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', disable=not verbose)
        for batch in pbar:
            x, labels = batch
            x = x.to(device)
            
            # One-hot encode labels
            c = torch.zeros(len(labels), num_classes).to(device)
            c[torch.arange(len(labels)), labels] = 1
            
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x, c)
            loss, recon_loss, kld = cvae_loss(recon_x, x, mu, logvar, beta)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_kld += kld.item()
            
            pbar.set_postfix({'loss': loss.item() / len(x)})
        
        train_loss /= len(train_loader.dataset)
        train_recon_loss /= len(train_loader.dataset)
        train_kld /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0
        val_recon_loss = 0
        val_kld = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x, labels = batch
                x = x.to(device)
                
                c = torch.zeros(len(labels), num_classes).to(device)
                c[torch.arange(len(labels)), labels] = 1
                
                recon_x, mu, logvar = model(x, c)
                loss, recon_loss, kld = cvae_loss(recon_x, x, mu, logvar, beta)
                
                val_loss += loss.item()
                val_recon_loss += recon_loss.item()
                val_kld += kld.item()
        
        val_loss /= len(val_loader.dataset)
        val_recon_loss /= len(val_loader.dataset)
        val_kld /= len(val_loader.dataset)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_recon_loss'].append(train_recon_loss)
        history['val_recon_loss'].append(val_recon_loss)
        history['train_kld'].append(train_kld)
        history['val_kld'].append(val_kld)
        
        if verbose:
            print(f'Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')
    
    return model, history


def get_latent_representations(model, data_loader, device, is_conditional=False, num_classes=None):
    """
    Extract latent representations from trained model
    
    Args:
        model: Trained VAE model
        data_loader: Data loader
        device: Device
        is_conditional: Whether the model is conditional
        num_classes: Number of classes (for conditional models)
    
    Returns:
        latent_features, labels
    """
    model.eval()
    latent_features = []
    labels_list = []
    
    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x, labels = batch
                labels_list.extend(labels.numpy())
            else:
                x = batch
            
            x = x.to(device) #type: ignore
            
            if is_conditional:
                # Create one-hot encoded condition
                c = torch.zeros(len(labels), num_classes).to(device)  #type: ignore
                c[torch.arange(len(labels)), labels] = 1
                z = model.get_latent(x, c)
            else:
                z = model.get_latent(x)
            
            latent_features.append(z.cpu().numpy())
    
    latent_features = np.concatenate(latent_features, axis=0)
    labels_array = np.array(labels_list) if labels_list else None
    
    return latent_features, labels_array


def get_reconstructions(model, data_loader, device, n_samples=5):
    """
    Get original and reconstructed samples for comparison
    """
    model.eval()
    
    originals = []
    reconstructions = []
    
    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x, _ = batch
            else:
                x = batch
            
            x = x.to(device) #type: ignore
            recon_x, _, _ = model(x)
            
            originals.append(x.cpu().numpy())
            reconstructions.append(recon_x.cpu().numpy())
            
            if len(originals) * len(x) >= n_samples:
                break
    
    originals = np.concatenate(originals, axis=0)[:n_samples]
    reconstructions = np.concatenate(reconstructions, axis=0)[:n_samples]
    
    return originals, reconstructions
