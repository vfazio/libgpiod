// SPDX-License-Identifier: LGPL-2.1-or-later
// SPDX-FileCopyrightText: 2022 Bartosz Golaszewski <brgl@bgdev.pl>
// SPDX-FileCopyrightText: 2024 Bartosz Golaszewski <bartosz.golaszewski@linaro.org>

#include <Python.h>
#include <linux/version.h>
#include <sys/prctl.h>
#include <sys/utsname.h>

/* Backport of standard macro available in Python 3.11 */
#if PY_VERSION_HEX < 0x030B0000
#define _PyCFunction_CAST(func) ((PyCFunction)(void(*)(void))(func))
#endif

/* Copied from gpiod/ext/common.c */
static unsigned int Py_gpiod_PyLongAsUnsignedInt(PyObject *pylong)
{
	unsigned long tmp;

	tmp = PyLong_AsUnsignedLong(pylong);
	if (PyErr_Occurred())
		return 0;

	if (tmp > UINT_MAX) {
		PyErr_SetString(PyExc_ValueError, "value exceeding UINT_MAX");
		return 0;
	}

	return tmp;
}

static PyObject *
module_set_process_name(PyObject *Py_UNUSED(self), PyObject *arg)
{
	const char *name;
	int ret;

	name = PyUnicode_AsUTF8(arg);
	if (!name)
		return NULL;

	ret = prctl(PR_SET_NAME, name);
	if (ret)
		return PyErr_SetFromErrno(PyExc_OSError);

	Py_RETURN_NONE;
}

static PyObject *
module_check_kernel_version(PyObject *Py_UNUSED(self), PyObject *const *args,
			    Py_ssize_t nargs)
{
	unsigned int req_maj, req_min, req_rel, curr_maj, curr_min, curr_rel;
	struct utsname un;
	int ret;

	if (nargs != 3)
		return PyErr_Format(PyExc_TypeError,
				    "check_kernel_version called with %ld arguments",
				    nargs);


	req_maj = Py_gpiod_PyLongAsUnsignedInt(args[0]);
	if (PyErr_Occurred())
		return NULL;
	req_min = Py_gpiod_PyLongAsUnsignedInt(args[1]);
	if (PyErr_Occurred())
		return NULL;
	req_rel = Py_gpiod_PyLongAsUnsignedInt(args[2]);
	if (PyErr_Occurred())
		return NULL;

	ret = uname(&un);
	if (ret)
		return PyErr_SetFromErrno(PyExc_OSError);

	ret = sscanf(un.release, "%u.%u.%u", &curr_maj, &curr_min, &curr_rel);
	if (ret != 3) {
		PyErr_SetString(PyExc_RuntimeError,
				"invalid linux version read from the kernel");
		return NULL;
	}

	if (KERNEL_VERSION(curr_maj, curr_min, curr_rel) <
	    KERNEL_VERSION(req_maj, req_min, req_rel))
		Py_RETURN_FALSE;

	Py_RETURN_TRUE;
}

static PyMethodDef module_methods[] = {
	{
		.ml_name = "set_process_name",
		.ml_meth = (PyCFunction)module_set_process_name,
		.ml_flags = METH_O,
	},
	{
		.ml_name = "check_kernel_version",
		.ml_meth = _PyCFunction_CAST(module_check_kernel_version),
		.ml_flags = METH_FASTCALL,
	},
	{ }
};

static struct PyModuleDef_Slot module_slots[] = {
	{0, NULL},
};

static PyModuleDef module_def = {
	PyModuleDef_HEAD_INIT,
	.m_name = "system._ext",
	.m_methods = module_methods,
	.m_slots = module_slots,
};

PyMODINIT_FUNC PyInit__ext(void)
{
	return PyModuleDef_Init(&module_def);
}
