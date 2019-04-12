#include <iostream>
#include <Python.h>

using namespace std;

int main(int argc, char *argv[])
{
//not useful
////    wchar_t *program = Py_DecodeLocale(argv[0], NULL);
////    char* python_home_ = (char*) "/usr";
////    char* program_name_ = (char*) "/usr/bin/python3.5";

////    Py_SetPythonHome(python_home_);
////    Py_SetProgramName(program_name_);
    float test_fl = 0;
    float test_fl1 = 0;
    float test_fl2 = 0;

    Py_Initialize();   //initialize python enviroment

    if(!Py_IsInitialized())
    {
        return -1;
    }
//temporary debug
//    printf("python home: %s\n", Py_GetPythonHome());
//    printf("program name: %s\n", Py_GetProgramName());
//    printf("Platform: %s\n", Py_GetPlatform());
//    printf("Version: %s\n", Py_GetVersion());

    PyObject* pModule = NULL;
    PyObject* pModule2 = NULL;
    PyObject* pFunc = NULL;
    PyObject* pParam = NULL;
    PyObject* pRet = NULL;

    PyObject* pFunc2 = NULL;
    PyObject* pParam2 = NULL;
    PyObject* pRet2 = NULL;
//    PyObject* pResult = NULL;
//    const char* pBuffer = NULL;
//    int iBufferSize = 0;

//    PyRun_SimpleString("print('python start:')");  //use python3 directly
//    PyRun_SimpleString("import cv2\n"
//                           "print(cv2.__version__)\n");
//    PyRun_SimpleString("import tensorflow as tf\n"
//                           "print(tf.__version__)\n");

    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('/home/why/doyle_why/language/python_work/c_python_test1/src/pythonpacktest/src')");

//    PyRun_SimpleString("sys.path.append('/usr/lib/python35.zip')");
//    PyRun_SimpleString("sys.path.append('/usr/lib/python3.5')");
//    PyRun_SimpleString("sys.path.append('/usr/lib/python3.5/plat-x86_64-linux-gnu')");
//    PyRun_SimpleString("sys.path.append('/usr/lib/python3.5/lib-dynload')");
//    PyRun_SimpleString("sys.path.append('/home/why/.local/lib/python3.5/site-packages')");
//    PyRun_SimpleString("sys.path.append('/usr/local/lib/python3.5/dist-packages')");
//    PyRun_SimpleString("sys.path.append('/usr/lib/python3/dist-packages')");

//    PyRun_SimpleString("print(sys.path)");

//    pModule = PyImport_ImportModule("pythontest1");
//    pModule2 = PyImport_ImportModule("tf_test1");
    pModule = PyImport_ImportModule("real_test2");

    if (!pModule)
    {
        cout << "get module from real_test2.py failed!" << endl;
        Py_Finalize();
        exit (0);
    }

    //test function 1
    pFunc = PyObject_GetAttrString(pModule, "train_angle");
    if (!pFunc)
    {
        cout << "get func failed!" << endl;
//        cout << int(pFunc) << endl;
        Py_Finalize();
        exit (0);
    }

    pParam = Py_BuildValue("(f,f,f)", 0.5, 0.6, 0.7);
    pRet = PyEval_CallObject(pFunc, pParam);

    PyArg_ParseTuple(pRet,"fff",&test_fl,&test_fl1,&test_fl2);//转换返回类型
    cout << "train_angle1:" << test_fl << endl;//输出结果
    cout << "train_angle2:" << test_fl1 << endl;//输出结果
    cout << "train_angle3:" << test_fl2 << endl;//输出结果


    //test function 2
    pFunc2 = PyObject_GetAttrString(pModule, "jacobian_angle");
    if (!pFunc2)
    {
        cout << "get func2 failed!" << endl;
        Py_Finalize();
        exit (0);
    }

    pParam2 = Py_BuildValue("(f,f,f,f,f,f)", 0.5, 0.6, 0.7, 1.5, 1.5, 1.5);
    pRet2 = PyEval_CallObject(pFunc2, pParam2);

    PyArg_ParseTuple(pRet2,"fff",&test_fl,&test_fl1,&test_fl2);//转换返回类型
    cout << "jacobian_angle1:" << test_fl << endl;//输出结果
    cout << "jacobian_angle2:" << test_fl1 << endl;//输出结果
    cout << "jacobian_angle3:" << test_fl2 << endl;//输出结果

   //not useful
//    pResult = PyEval_CallObject(pFunc,pParam);
//    if(pResult)
//    {
//        if(PyArg_Parse(pResult, "(si)", &pBuffer, iBufferSize))
//        {
//            cout << pBuffer << endl;
//            cout << iBufferSize << endl;
//        }
//    }
//    Py_DECREF(pParam);
//    Py_DECREF(pFunc);

    Py_Finalize();


    cout << "happy ending!" << endl;

    return 0;

}
