using System;
using System.Runtime.InteropServices;

public class PyLauncher {
    [DllImport(@"C:\Users\vinay\AppData\Local\Programs\Python\Python312\python312.dll", CharSet = CharSet.Unicode)]
    public static extern int Py_Main(int argc, string[] argv);

    public static int Main(string[] args) {
        string[] allArgs = new string[args.Length + 1];
        allArgs[0] = "python";
        for (int i = 0; i < args.Length; i++) {
            allArgs[i + 1] = args[i];
        }
        return Py_Main(allArgs.Length, allArgs);
    }
}
