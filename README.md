# Projet GAN - Front / Back separes

## Structure

- `tp/back` : API FastAPI qui charge `gan_generator_cats.pt` et genere une image.
- `tp/front` : interface PHP simple avec un bouton pour demander une generation.

## Prerequis

- Environnement virtuel deja cree a la racine : `.venv`
- Dependances installees depuis `requirements.txt`
- PHP installe (pour le front)

## Lancer le backend (FastAPI)

Depuis la racine du projet :

```powershell
.\.venv\Scripts\python.exe -m uvicorn tp.back.app.main:app --reload --host 127.0.0.1 --port 8000
```

Endpoints utiles :

- `GET http://127.0.0.1:8000/health`
- `POST http://127.0.0.1:8000/generate`

## Lancer le frontend (PHP)

Dans un second terminal, depuis `tp/front` :

```powershell
php -S 127.0.0.1:8080
```

Puis ouvrir :

- `http://127.0.0.1:8080`

## Notes

- Le backend utilise le checkpoint `gan_generator_cats.pt` situe a la racine.
- Si l'API renvoie une erreur 503, verifier que le modele est bien charge au demarrage.
