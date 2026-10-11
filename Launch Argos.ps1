Add-Type -AssemblyName PresentationFramework, PresentationCore, WindowsBase

[xml]$xaml = @'
<Window xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
        xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
        Title="Project Argos" Height="315" Width="405" ResizeMode="NoResize"
        WindowStartupLocation="CenterScreen" Background="#050C08" Foreground="#00FF66"
        FontFamily="Consolas">
  <Border Margin="13" Padding="20" Background="#002211" BorderBrush="#008F11" BorderThickness="1">
    <StackPanel>
      <TextBlock Text="ARGOS // LOCAL VISION" FontSize="20" FontWeight="Bold" Foreground="#00FF66" Margin="0,0,0,5"/>
      <TextBlock Text="TRACK DATABASE  /  OPTICAL SYSTEM" FontSize="11" Foreground="#00AA55" Margin="0,0,0,17"/>
      <TextBlock Text="SELECT CAMERA CONFIGURATION" FontSize="12" Foreground="#39FF14" Margin="0,0,0,7"/>
      <RadioButton x:Name="SingleMode" Content="SINGLE CAMERA" IsChecked="True" Foreground="#00FF66" Margin="0,4"/>
      <RadioButton x:Name="StereoMode" Content="STEREO CAMERAS" Foreground="#00FF66" Margin="0,4"/>
      <TextBlock x:Name="Status" Text="SYSTEM READY" Foreground="#39FF14" Margin="0,16,0,12"/>
      <StackPanel Orientation="Horizontal" HorizontalAlignment="Left">
        <Button x:Name="StartButton" Content="START ARGOS" Width="145" Height="36" Margin="0,0,10,0"
                Background="#00FF66" Foreground="#050C08" BorderBrush="#39FF14" FontWeight="Bold"/>
        <Button x:Name="StopButton" Content="STOP" Width="90" Height="36" IsEnabled="False"
                Background="#002211" Foreground="#39FF14" BorderBrush="#008F11"/>
      </StackPanel>
      <TextBlock Text="LOCAL SESSION  |  Q / ESC TO CLOSE FEED" FontSize="10" Foreground="#008F11" Margin="0,16,0,0"/>
    </StackPanel>
  </Border>
</Window>
'@

$reader = [System.Xml.XmlNodeReader]::new($xaml)
$window = [Windows.Markup.XamlReader]::Load($reader)
$single = $window.FindName('SingleMode')
$stereo = $window.FindName('StereoMode')
$status = $window.FindName('Status')
$start = $window.FindName('StartButton')
$stop = $window.FindName('StopButton')
$script:argosProcess = $null
$script:stoppingByUser = $false
$projectRoot = $PSScriptRoot
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
$logDirectory = Join-Path $projectRoot 'logs'
$stdoutPath = Join-Path $logDirectory 'launcher-stdout.log'
$stderrPath = Join-Path $logDirectory 'launcher-stderr.log'

function Show-ArgosFailure([string]$Message) {
    $details = @()
    foreach ($path in @($stderrPath, $stdoutPath)) {
        if (Test-Path -LiteralPath $path) {
            $content = Get-Content -LiteralPath $path -Tail 25 -ErrorAction SilentlyContinue
            if ($content) { $details += ($content -join [Environment]::NewLine) }
        }
    }
    if ($details.Count -gt 0) {
        $Message += [Environment]::NewLine + [Environment]::NewLine + ($details -join [Environment]::NewLine)
    }
    $Message += [Environment]::NewLine + [Environment]::NewLine + "Full logs: $logDirectory"
    [System.Windows.MessageBox]::Show($Message, 'Project Argos could not start') | Out-Null
}

$start.Add_Click({
    if (-not (Test-Path -LiteralPath $pythonExe)) {
        Show-ArgosFailure 'The project virtual environment is missing. Open a terminal in the Argos folder and run: py -m venv .venv, then .venv\Scripts\python.exe -m pip install -e ".[dev]".'
        return
    }
    $mode = if ($single.IsChecked) { 'single' } else { 'stereo' }
    try {
        New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
        Remove-Item -LiteralPath $stdoutPath, $stderrPath -Force -ErrorAction SilentlyContinue
        $script:stoppingByUser = $false
        $script:argosProcess = Start-Process -FilePath $pythonExe -ArgumentList @('-m', 'argos.app', '--mode', $mode) `
            -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
        $status.Text = "Running in $mode-camera mode"
        $start.IsEnabled = $false
        $stop.IsEnabled = $true
    } catch {
        Show-ArgosFailure $_.Exception.Message
    }
})

$stop.Add_Click({
    if ($script:argosProcess -and -not $script:argosProcess.HasExited) {
        $script:stoppingByUser = $true
        $script:argosProcess.Kill()
        $status.Text = 'Stopping…'
    }
})

$timer = [Windows.Threading.DispatcherTimer]::new()
$timer.Interval = [TimeSpan]::FromMilliseconds(400)
$timer.Add_Tick({
    if ($script:argosProcess -and $script:argosProcess.HasExited) {
        $script:argosProcess.Refresh()
        $exitCode = $script:argosProcess.ExitCode
        $script:argosProcess = $null
        $status.Text = if ($exitCode -eq 0) { 'Argos closed' } else { "Argos stopped with error ($exitCode)" }
        $start.IsEnabled = $true
        $stop.IsEnabled = $false
        if ($exitCode -ne 0 -and -not $script:stoppingByUser) {
            Show-ArgosFailure "Argos exited with code $exitCode. Check the error details below."
        }
        $script:stoppingByUser = $false
    }
})
$timer.Start()

$window.Add_Closed({
    $timer.Stop()
    if ($script:argosProcess -and -not $script:argosProcess.HasExited) {
        $script:argosProcess.Kill()
    }
})
[void]$window.ShowDialog()
