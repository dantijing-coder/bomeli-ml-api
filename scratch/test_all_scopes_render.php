<?php
/**
 * Test rendering of ai-sales-insights.php across all scopes and viewports.
 */
$_SESSION = [
    'user_id' => 1,
    'username' => 'admin',
    'role' => 'admin',
    'branch_id' => 0
];

$scopes = [
    ['month' => '2026-09', 'branch_id' => 0, 'name' => 'Consolidated Global Network'],
    ['month' => '2026-09', 'branch_id' => 1, 'name' => 'Lala Branch'],
    ['month' => '2026-09', 'branch_id' => 2, 'name' => 'Kapatagan Branch'],
    ['month' => '2026-09', 'branch_id' => 3, 'name' => 'Tubod Branch (Empty State)']
];

chdir(__DIR__ . '/../../pages');

foreach ($scopes as $s) {
    $_GET['month'] = $s['month'];
    $_GET['branch_id'] = $s['branch_id'];

    ob_start();
    try {
        include 'ai-sales-insights.php';
        $output = ob_get_clean();
        $len = strlen($output);
        $hasSpline = strpos($output, 'cubicInterpolationMode') !== false;
        $hasForecast = strpos($output, '3-Month Cash Inflow & Collection Forecast') !== false;
        $hasMaturing = strpos($output, 'matures') !== false || strpos($output, 'Target Run-Rate') !== false;

        echo "[RENDER OK] {$s['name']} — {$len} bytes (Spline={$hasSpline}, Forecast={$hasForecast})\n";
    } catch (Throwable $e) {
        ob_end_clean();
        echo "[RENDER ERROR] {$s['name']} — " . $e->getMessage() . "\n";
    }
}
