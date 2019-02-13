Prepare your data for SchNetPack
================================

Converting xyz-files to extended xyz
------------------------------------

In order to automatically transfer data to an ase database that can be used
to train the model, xyz files must be transformed into extended xyz. Therefore
a comment string with a definition of the atomic properties and the molecular
properties is required. The following snippet is taken from an xyz-file for
`ethanol <http://quantum-machine.org/gdml/>`_::

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

Usually xyz-files should start with the number of atoms in the first line and
a comment second. In this case the comment line contains the total energy of
the molecule. The following lines contain the atomic properties of the
molecule. In this case the first column contains the atom type, the following
three columns contain the positions and the last three columns consist of the
atomic forces. In order to convert this example to an extended xyz-file the
comment line must store the property information. Therefore the file must be
parsed and converted to the new format. First of all you need to define a
property string. The property string starts with ``Properties=`` and is
followed by teh column information with the shape
``column name:datatype:number of columns``. For the ethanol example above you
get::

    properties = 'Properties=species:S:1:pos:R:3:forces:R:3'

Secondly you need to open a new file with::

    new_file = open('./new_file.xyz', 'w')

and loop over all lines of the old file by using ``xyz_file.readline()``
until you end up getting blank lines. Every loop reads the number of atoms
and the comment line first::

    # inside the loop
    first_line = xyz_file.readline()
    comment = xyz_file.readline()
    n_atoms = int(first_line.strip('\n'))

In this case the comment only contains the energy, you extract it::

    energy = float(comment.strip('/n'))

.. note::

    Your xyz-file could contain other properties than the energy so they
    would need to be extracted differently. In the end they need to be
    appended to the property string.

As a next step the first line and the new comment are written to the new file::

    new_file.writelines(first_line)
    new_file.writelines(' '.join([properties, 'energy={}'.format(energy)]) +
    '\n')

The last step is to loop over the atoms and transfer all the atomic f
properties to the new file::

    for i in range(n_atoms):
        line = xyz_file.readline()
        new_file.writelines(line)

In the end you get a new file which contains the desired property string in
the comment line::

    9
    Properties=species:S:1:pos:R:3:forces:R:3 energy=-97208.40600498248
    C	0.0072	-0.5687	0.0	1.4517297437249999	6.01923561735	5.2067503058e-07
    C	-1.2854	0.2499	0.0	17.9533587683	-5.1623821924	3.49002255005e-07
    O	1.1304	0.3147	0.0	-4.0883694515	22.5901955842	3.30876819905e-06
    H	0.0392	-1.1972	0.89	-1.141572648295	-9.7469181345	7.64734244805
    H	0.0392	-1.1972	-0.89	-1.141572648295	-9.7469181345	-7.64734244805
    H	-1.3175	0.8784	0.89	-2.48205465155	4.933531278999999	4.3699824457
    H	-1.3175	0.8784	-0.89	-2.48205465155	4.933531278999999	-4.3699824457
    H	-2.1422	-0.4239	0.0	-5.5147904611	-3.0206752464999997	-8.9092739811e-09
    H	1.9857	-0.1365	0.0	-2.4392777023	-10.83820307755	-6.07213606025e-08

If your xyz-file has the same comment string as this example-file you can
simply use the ``extend_xyz()`` function from ``schnetpack.datasets.extxyz``
and define your property string. The function will then automatically
transform your file.

Transfering extended xyz-files to an ase database
-------------------------------------------------

SchNetPack generally requires the data as ase database. If the data is
already transformed to extended xyz-files is can directly be transformed to
an ase database by using ``parse_extxyz()`` from
``schnetpack.datasets.extxyz``::

    from schnetpack.datasets.extxyz import parse_extxyz

    parse_extxyz('path/to/db/dataset.db', 'path/to/file/xyz_file.xyz')

This will create a new ase database located at *path/to/db*.

.. note::

    The ``parse_extxyz()`` function should only be used for datasets that
    have **energy** and **forces** properties. Other properties will not be
    transferred to the database.

Using SchNetPack with extended xyz-files
----------------------------------------

Instead of converting your data manually to an ase database, you can also use
the ``ExtXYZ`` dataset class from ``schnetpack.datasets.extxyz``. This will
automatically create and use the ase database file.

Using SchNetPack with ase databases
-----------------------------------

In case your data is already formatted as an ase database use the
``AtomsData`` class from ``schnetpack.data``. The dataset requires a path to
the database and a definition of the molecular properties that are contained.
If the database has automatically been created from an xyz file, the
required properties are **energy** and **forces**::

    from schnetpack.data import AtomsData

    properties = ['energy', 'forces']
    dataset = AtomsData('path/to/db/database.db', properties=properties)

Using SchNetPack with pre-implemented datasets
----------------------------------------------

SchNetPack comes with several implementations of datasets, which are
downloaded automatically. This includes *ANI1*, *ISO17*, *Materials Project*,
*MD17*, *Organic Materials Database* and *QM9*. In order to use these
datasets select the proper dataset class from ``schnetpack.datasets`` and
define the path to the database. If the database does not exist at the
defined location, it will be downloaded automatically. If no properties are
passed to the dataset, all available properties will be used.
