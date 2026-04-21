<?php
declare(strict_types=1);

$apiUrl = 'http://127.0.0.1:8000/generate';
$error = null;
$imageDataUri = null;

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $ch = curl_init($apiUrl);
    if ($ch === false) {
        $error = "Impossible d'initialiser cURL.";
    } else {
        curl_setopt_array($ch, [
            CURLOPT_POST => true,
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_HTTPHEADER => ['Content-Type: application/json'],
            CURLOPT_POSTFIELDS => '{}',
            CURLOPT_TIMEOUT => 60,
        ]);

        $response = curl_exec($ch);
        $statusCode = (int) curl_getinfo($ch, CURLINFO_HTTP_CODE);

        if ($response === false) {
            $error = 'Erreur API: ' . curl_error($ch);
        } elseif ($statusCode !== 200) {
            $error = "Erreur API (HTTP $statusCode).";
        } else {
            $decoded = json_decode($response, true);
            if (!is_array($decoded) || !isset($decoded['image_base64']) || !is_string($decoded['image_base64'])) {
                $error = 'Reponse API invalide.';
            } else {
                $imageDataUri = 'data:image/png;base64,' . $decoded['image_base64'];
            }
        }
        curl_close($ch);
    }
}
?>
<!doctype html>
<html lang="fr">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Generateur de visages GAN</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <main class="container">
        <h1>Generateur de visages GAN</h1>
        <p>Cliquer sur le bouton pour generer un nouveau visage artificiel.</p>

        <form method="post">
            <button type="submit">Generer un visage</button>
        </form>

        <?php if ($error !== null): ?>
            <p class="error"><?= htmlspecialchars($error, ENT_QUOTES, 'UTF-8') ?></p>
        <?php endif; ?>

        <?php if ($imageDataUri !== null): ?>
            <div class="image-wrapper">
                <img src="<?= htmlspecialchars($imageDataUri, ENT_QUOTES, 'UTF-8') ?>" alt="Visage genere par GAN">
            </div>
        <?php endif; ?>
    </main>
</body>
</html>
