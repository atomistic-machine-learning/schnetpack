.. _Prepare Data:

Prepare your data for SchNetPack
********************************

SchNetPack requires your data to be provided as ``ase`` database. If your
data consists of ``.xyz`` or ``.extxyz`` files, you can use the parsing
script in order to convert your data. The script is called with::

    $ spk_parse.py parse with file_path=<file-path> db_path=tutorials/ethanol.db

The script will automatically check your file extension in order to
distinguish between ``.xyz`` and ``.extxyz`` files. If you provide an ``.xyz``
file, make sure to define the right molecular properties and if your
data contains forces. The following example shows a snippet of an ``.xyz``
file for `ethanol <http://quantum-machine.org/gdml/>`_::

    9
    -97208.40600498248
    C	0.0072	-0.5687	0.0	1.4517297437249999	6.01923561735	5.2067503058e-07
    C	-1.2854	0.2499	0.0	17.9533587683	-5.1623821924	3.49002255005e-07
    O	1.1304	0.3147	0.0	-4.0883694515	22.5901955842	3.30876819905e-06
    H	0.0392	-1.1972	0.89	-1.141572648295	-9.7469181345	7.64734244805
    H	0.0392	-1.1972	-0.89	-1.141572648295	-9.7469181345	-7.64734244805
    H	-1.3175	0.8784	0.89	-2.48205465155	4.933531278999999	4.3699824457
    H	-1.3175	0.8784	-0.89	-2.48205465155	4.933531278999999	-4.3699824457
    H	-2.1422	-0.4239	0.0	-5.5147904611	-3.0206752464999997	-8.9092739811e-09
    H	1.9857	-0.1365	0.0	-2.4392777023	-10.83820307755	-6.07213606025e-08

The data contains ``molecular_properties`` in the second row and atomic
properties in the following rows. All files contain at least the atom types and
the atomic positions in the first columns. Some files may also contain atomic
forces. In this case add the ``forces`` flag to your run arguments. In order
to parse the ethanol snippet run::


    $ spk_parse.py parse with forces file_path=<file-path> db_path=<db-path>
      "molecular_properties=['energy']"

.. note::

    Your xyz-file could contain other molecular properties than just energy so
    you would need to add them to the ``molecular_properties`` list. The default
    settings for molecular properties is **energy**.

If you provide your data as an ``.extxyz`` file, all necessary information is
provided within your data. You do not need to define any molecular properties
or add a forces tag.
