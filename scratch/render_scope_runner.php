<?php
$_SESSION = [
    'user_id' => 1,
    'username' => 'admin',
    'role' => 'admin',
    'branch_id' => 0
];

$_GET['month'] = $argv[1] ?? '2026-09';
$_GET['branch_id'] = isset($argv[2]) ? (int)$argv[2] : 0;

chdir(__DIR__ . '/../../pages');
include 'ai-sales-insights.php';
