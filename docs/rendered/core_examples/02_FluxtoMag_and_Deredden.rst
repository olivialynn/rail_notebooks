Flux to Mag And Deredden
========================

author: Sam Schmidt

last successfully run: Apr 26, 2023

.. code:: ipython3

    import matplotlib.pyplot as plt
    import os
    import tables_io
    import tempfile
    from rail.core.stage import RailStage
    from rail.core.data import TableHandle
    from rail.utils.path_utils import find_rail_file
    from rail.tools.photometry_tools import LSSTFluxToMagConverter, Dereddener

.. code:: ipython3

    DS = RailStage.data_store
    example_file = find_rail_file("examples_data/testdata/rubin_dm_dc2_example.pq")
    test_data = DS.read_file("test_data", TableHandle, example_file)


.. parsed-literal::

    column_list None


.. code:: ipython3

    test_data().info()


.. parsed-literal::

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 100 entries, 0 to 99
    Data columns (total 15 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   objectId          100 non-null    int64  
     1   ra                100 non-null    float64
     2   decl              100 non-null    float64
     3   u_gaap1p0Flux     100 non-null    float64
     4   u_gaap1p0FluxErr  100 non-null    float64
     5   g_gaap1p0Flux     100 non-null    float64
     6   g_gaap1p0FluxErr  100 non-null    float64
     7   r_gaap1p0Flux     100 non-null    float64
     8   r_gaap1p0FluxErr  100 non-null    float64
     9   i_gaap1p0Flux     100 non-null    float64
     10  i_gaap1p0FluxErr  100 non-null    float64
     11  z_gaap1p0Flux     100 non-null    float64
     12  z_gaap1p0FluxErr  100 non-null    float64
     13  y_gaap1p0Flux     100 non-null    float64
     14  y_gaap1p0FluxErr  100 non-null    float64
    dtypes: float64(14), int64(1)
    memory usage: 11.8 KB


Fluxes to Mags
~~~~~~~~~~~~~~

To convert fluxes to mags, we need to specify patterns for the
``flux_name`` and ``flux_err_name`` columns to be converted, and the
``mag_name`` and ``mag_err_name`` columns that will store the newly
created magnitudes.

This is done as below, by specifying a string listing the bands, and
``{band}`` in the patterns where the individual bands will go. The
dictionary below duplicates the default behavior of the converter, but
is written out explicitly as an example:

.. code:: ipython3

    # convert "gaap" fluxes to magnitudes:
    ftomagdict = dict(bands = "ugrizy",
                      flux_name="{band}_gaap1p0Flux",
                      flux_err_name="{band}_gaap1p0FluxErr",
                      mag_name="mag_{band}_lsst",
                      mag_err_name="mag_err_{band}_lsst",
                      copy_cols=dict(ra='ra', dec='decl', objectId='objectId'))
    fluxToMag = LSSTFluxToMagConverter.make_stage(name='flux2mag', **ftomagdict)

.. code:: ipython3

    mags_data = fluxToMag(test_data)


.. parsed-literal::

    Inserting handle into data store.  output_flux2mag: inprogress_output_flux2mag.pq, flux2mag


.. parsed-literal::

    /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages/rail/tools/photometry_tools.py:377: RuntimeWarning: invalid value encountered in log10
      vals = -2.5*np.log10(flux_vals) + self.config.mag_offset


.. code:: ipython3

    list(mags_data().keys())




.. parsed-literal::

    ['mag_u_lsst',
     'mag_err_u_lsst',
     'mag_g_lsst',
     'mag_err_g_lsst',
     'mag_r_lsst',
     'mag_err_r_lsst',
     'mag_i_lsst',
     'mag_err_i_lsst',
     'mag_z_lsst',
     'mag_err_z_lsst',
     'mag_y_lsst',
     'mag_err_y_lsst',
     'ra',
     'dec',
     'objectId']



Deredden Mags
~~~~~~~~~~~~~

To deredden magnitudes we need to grab one of the dust maps used by the
``dustmaps`` package. We’ll grab the default Schlegel-Finkbeiner-Davis
“SFD” map. NOTE: This will download a file to your machine containing
the SFD data!

We need to feed the location of the directory containing the newly
created “sfd” maps to the stage. As we downloaded the data to the
present working directory with the command above, that directory is just
``"./"``

.. code:: ipython3

    dustmap_dir = "./"
    
    dereddener = Dereddener.make_stage(name='dereddener', dustmap_dir=dustmap_dir)
    dereddener.fetch_map()


.. parsed-literal::

    Downloading SFD data file to /home/runner/work/rail_notebooks/rail_notebooks/rail/examples/core_examples/sfd/SFD_dust_4096_ngp.fits


.. parsed-literal::

    /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages/dustmaps/config.py:74: ConfigWarning: Configuration file not found:
    
        /home/runner/.dustmapsrc
    
    To create a new configuration file in the default location, run the following python code:
    
        from dustmaps.config import config
        config.reset()
    
    Note that this will delete your configuration! For example, if you have specified a data directory, then dustmaps will forget about its location.
      warn(('Configuration file not found:\n\n'


.. parsed-literal::

    Downloading data to '/home/runner/work/rail_notebooks/rail_notebooks/rail/examples/core_examples/sfd/SFD_dust_4096_ngp.fits' ...
    Downloading https://dataverse.harvard.edu/api/access/datafile/2902687 ...


.. parsed-literal::

      0.0 B of 64.0 MiB |   0.0 s/B |                       | [38;2;255;0;0m  0%[39m | ETA:  --:--:--

.. parsed-literal::

    830.0 KiB of 64.0 MiB |   8.6 MiB/s |                   | [38;2;255;8;0m  1%[39m | ETA:  00:00:00

.. parsed-literal::

      4.1 MiB of 64.0 MiB |  26.8 MiB/s |#                  | [38;2;255;42;0m  6%[39m | ETA:   0:00:02

.. parsed-literal::

      8.1 MiB of 64.0 MiB |  26.8 MiB/s |##                 | [38;2;255;90;0m 12%[39m | ETA:   0:00:02

.. parsed-literal::

     12.2 MiB of 64.0 MiB |  44.7 MiB/s |###                | [38;2;255;108;0m 18%[39m | ETA:   0:00:01

.. parsed-literal::

     16.2 MiB of 64.0 MiB |  44.7 MiB/s |####               | [38;2;255;125;0m 25%[39m | ETA:   0:00:01

.. parsed-literal::

     20.3 MiB of 64.0 MiB |  51.0 MiB/s |######             | [38;2;255;143;0m 31%[39m | ETA:   0:00:00

.. parsed-literal::

     23.5 MiB of 64.0 MiB |  51.0 MiB/s |######             | [38;2;255;157;0m 36%[39m | ETA:   0:00:00

.. parsed-literal::

     26.7 MiB of 64.0 MiB |  53.7 MiB/s |#######            | [38;2;255;168;0m 41%[39m | ETA:   0:00:00

.. parsed-literal::

     30.8 MiB of 64.0 MiB |  53.7 MiB/s |#########          | [38;2;255;204;0m 48%[39m | ETA:   0:00:00

.. parsed-literal::

     34.0 MiB of 64.0 MiB |  55.8 MiB/s |##########         | [38;2;255;232;0m 53%[39m | ETA:   0:00:00

.. parsed-literal::

     38.1 MiB of 64.0 MiB |  55.8 MiB/s |###########        | [38;2;248;255;0m 59%[39m | ETA:   0:00:00

.. parsed-literal::

     42.1 MiB of 64.0 MiB |  57.4 MiB/s |############       | [38;2;230;255;0m 65%[39m | ETA:   0:00:00

.. parsed-literal::

     45.4 MiB of 64.0 MiB |  57.4 MiB/s |#############      | [38;2;216;255;0m 70%[39m | ETA:   0:00:00

.. parsed-literal::

     48.6 MiB of 64.0 MiB |  57.4 MiB/s |##############     | [38;2;202;255;0m 75%[39m | ETA:   0:00:00

.. parsed-literal::

     51.9 MiB of 64.0 MiB |  57.4 MiB/s |###############    | [38;2;188;255;0m 81%[39m | ETA:   0:00:00

.. parsed-literal::

     55.9 MiB of 64.0 MiB |  58.4 MiB/s |################   | [38;2;170;255;0m 87%[39m | ETA:   0:00:00

.. parsed-literal::

     60.0 MiB of 64.0 MiB |  58.4 MiB/s |#################  | [38;2;77;255;0m 93%[39m | ETA:   0:00:00

.. parsed-literal::

     64.0 MiB of 64.0 MiB |  59.4 MiB/s |###################| [38;2;0;255;0m100%[39m | ETA:  00:00:00

.. parsed-literal::

    Downloading SFD data file to /home/runner/work/rail_notebooks/rail_notebooks/rail/examples/core_examples/sfd/SFD_dust_4096_sgp.fits


.. parsed-literal::

    Downloading data to '/home/runner/work/rail_notebooks/rail_notebooks/rail/examples/core_examples/sfd/SFD_dust_4096_sgp.fits' ...
    Downloading https://dataverse.harvard.edu/api/access/datafile/2902695 ...


.. parsed-literal::

      0.0 B of 64.0 MiB |   0.0 s/B |                       | [38;2;255;0;0m  0%[39m | ETA:  --:--:--

.. parsed-literal::

    717.0 KiB of 64.0 MiB |   7.0 MiB/s |                   | [38;2;255;7;0m  1%[39m | ETA:   0:00:09

.. parsed-literal::

      3.2 MiB of 64.0 MiB |   7.0 MiB/s |                   | [38;2;255;33;0m  5%[39m | ETA:   0:00:08

.. parsed-literal::

      7.3 MiB of 64.0 MiB |  34.9 MiB/s |##                 | [38;2;255;86;0m 11%[39m | ETA:   0:00:01

.. parsed-literal::

     11.3 MiB of 64.0 MiB |  34.9 MiB/s |###                | [38;2;255;104;0m 17%[39m | ETA:   0:00:01

.. parsed-literal::

     15.4 MiB of 64.0 MiB |  47.7 MiB/s |####               | [38;2;255;122;0m 24%[39m | ETA:   0:00:01

.. parsed-literal::

     19.4 MiB of 64.0 MiB |  47.7 MiB/s |#####              | [38;2;255;140;0m 30%[39m | ETA:   0:00:00

.. parsed-literal::

     23.5 MiB of 64.0 MiB |  53.7 MiB/s |######             | [38;2;255;157;0m 36%[39m | ETA:   0:00:00

.. parsed-literal::

     27.5 MiB of 64.0 MiB |  53.7 MiB/s |########           | [38;2;255;176;0m 43%[39m | ETA:   0:00:00

.. parsed-literal::

     31.6 MiB of 64.0 MiB |  57.3 MiB/s |#########          | [38;2;255;211;0m 49%[39m | ETA:   0:00:00

.. parsed-literal::

     35.6 MiB of 64.0 MiB |  57.3 MiB/s |##########         | [38;2;255;246;0m 55%[39m | ETA:   0:00:00

.. parsed-literal::

     39.7 MiB of 64.0 MiB |  59.5 MiB/s |###########        | [38;2;241;255;0m 62%[39m | ETA:   0:00:00

.. parsed-literal::

     42.1 MiB of 64.0 MiB |  59.5 MiB/s |############       | [38;2;230;255;0m 65%[39m | ETA:   0:00:00

.. parsed-literal::

     46.2 MiB of 64.0 MiB |  59.5 MiB/s |#############      | [38;2;212;255;0m 72%[39m | ETA:   0:00:00

.. parsed-literal::

     50.2 MiB of 64.0 MiB |  59.5 MiB/s |##############     | [38;2;195;255;0m 78%[39m | ETA:   0:00:00

.. parsed-literal::

     54.3 MiB of 64.0 MiB |  60.9 MiB/s |################   | [38;2;177;255;0m 84%[39m | ETA:   0:00:00

.. parsed-literal::

     58.3 MiB of 64.0 MiB |  60.9 MiB/s |#################  | [38;2;159;255;0m 91%[39m | ETA:   0:00:00

.. parsed-literal::

     62.4 MiB of 64.0 MiB |  61.9 MiB/s |################## | [38;2;30;255;0m 97%[39m | ETA:   0:00:00

.. code:: ipython3

    deredden_data = dereddener(mags_data)


.. parsed-literal::

    Inserting handle into data store.  output_dereddener: inprogress_output_dereddener.pq, dereddener


.. code:: ipython3

    deredden_data().keys()




.. parsed-literal::

    Index(['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst', 'mag_z_lsst',
           'mag_y_lsst'],
          dtype='object')



We see that the deredden stage returns us a dictionary with the
dereddened magnitudes. Let’s plot the difference of the un-dereddened
magnitudes and the dereddened ones for u-band to see if they are,
indeed, slightly brighter:

.. code:: ipython3

    delta_u_mag = mags_data()['mag_u_lsst'] - deredden_data()['mag_u_lsst']
    plt.figure(figsize=(8,6))
    plt.scatter(mags_data()['mag_u_lsst'], delta_u_mag, s=15)
    plt.xlabel("orignal u-band mag", fontsize=12)
    plt.ylabel("u - deredden_u");



.. image:: ../../../docs/rendered/core_examples/02_FluxtoMag_and_Deredden_files/../../../docs/rendered/core_examples/02_FluxtoMag_and_Deredden_14_0.png


Clean up
~~~~~~~~

For cleanup, uncomment the line below to delete that SFD map directory
downloaded in this example:

.. code:: ipython3

    #! rm -rf sfd/
