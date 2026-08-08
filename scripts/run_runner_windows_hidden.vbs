Option Explicit

If WScript.Arguments.Count <> 4 Then
    WScript.Quit 2
End If

Function QuoteArg(value)
    QuoteArg = Chr(34) & Replace(CStr(value), Chr(34), Chr(34) & Chr(34)) & Chr(34)
End Function

Dim shell, powerShell, command, exitCode
Set shell = CreateObject("WScript.Shell")
powerShell = shell.ExpandEnvironmentStrings("%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe")

command = QuoteArg(powerShell) _
    & " -NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden" _
    & " -File " & QuoteArg(WScript.Arguments(0)) _
    & " -ConfigPath " & QuoteArg(WScript.Arguments(1)) _
    & " -TokenPath " & QuoteArg(WScript.Arguments(2)) _
    & " -LogFile " & QuoteArg(WScript.Arguments(3))

exitCode = shell.Run(command, 0, True)
WScript.Quit exitCode
