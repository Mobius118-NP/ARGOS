@echo off
start "" powershell.exe -NoProfile -ExecutionPolicy Bypass -STA -WindowStyle Hidden -Command "$ErrorActionPreference='Stop'; try { & '%~dp0Launch Argos.ps1' } catch { Add-Type -AssemblyName System.Windows.Forms; [void][System.Windows.Forms.MessageBox]::Show($_.Exception.ToString(), 'Project Argos launcher error') }"
