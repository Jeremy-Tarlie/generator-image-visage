# Entrainement modeles visages

Ce dossier entraine des modeles generatifs sur les images presentes dans `data/`.

## 1) VAE (plus stable, moins net)

Depuis la racine du projet :

```powershell
.\.venv\Scripts\python.exe .\tp\back\generation_model\train_vae_faces.py --epochs 60 --resume
```

Sorties :

- `vae_faces.pt` : checkpoint du modele (a la racine de `generation_model`)
- `vae_faces_samples.png` : generations aleatoires
- `vae_faces_reconstructions.png` : comparaison original/reconstruction

## 2) GAN (plus net, recommande pour realisme)

Depuis la racine du projet :

```powershell
.\.venv\Scripts\python.exe .\tp\back\generation_model\train_gan_faces.py --epochs 80
```

Pour reprendre :

```powershell
.\.venv\Scripts\python.exe .\tp\back\generation_model\train_gan_faces.py --epochs 80 --resume
```

Sorties :

- `gan_faces.pt` : checkpoint GAN
- `gan_faces_samples.png` : echantillons GAN

## Important

- Pour des visages plus propres, le GAN est generalement meilleur que le VAE.
- Sur CPU, l'entrainement est lent. Si tu as un GPU CUDA, la qualite monte plus vite car tu peux entrainer plus longtemps.
