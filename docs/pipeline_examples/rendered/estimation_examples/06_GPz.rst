GPz Estimation Example
======================

**Author:** Sam Schmidt

**Last Run Successfully:** September 26, 2023

**Note:** If you’re planning to run this in a notebook, you may want to
use interactive mode instead. See
```GPz.ipynb`` <https://github.com/LSSTDESC/rail/blob/main/interactive_examples/estimation_examples/GPz.ipynb>`__
in the ``interactive_examples/estimation_examples/`` folder for a
version of this notebook in interactive mode.

A quick demo of running GPz on the typical test data. You should have
installed rail_gpz_v1 (we highly recommend that you do this from within
a custom conda environment so that all dependencies for package versions
are met), either by cloning and installing from github, or with:

::

   pip install pz-rail-gpz-v1

As RAIL is a namespace package, installing rail_gpz_v1 will make
``GPzInformer`` and ``GPzEstimator`` available, and they can be imported
via:

::

   from rail.estimation.algos.gpz import GPzInformer, GPzEstimator

Let’s start with all of our necessary imports:

.. code:: ipython3

    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import rail
    import tables_io
    import qp
    from rail.core.data import TableHandle
    from rail.core.stage import RailStage
    from rail.estimation.algos.gpz import GPzInformer, GPzEstimator

.. code:: ipython3

    # find_rail_file is a convenience function that finds a file in the RAIL ecosystem   We have several example data files that are copied with RAIL that we can use for our example run, let's grab those files, one for training/validation, and the other for testing:
    from rail.utils.path_utils import find_rail_file
    trainFile = find_rail_file('examples_data/testdata/test_dc2_training_9816.hdf5')
    testFile = find_rail_file('examples_data/testdata/test_dc2_validation_9816.hdf5')
    training_data = tables_io.read(trainFile)
    test_data = tables_io.read(testFile)

Now, we need to set up the stage that will run GPz. We begin by defining
a dictionary with the config options for the algorithm. There are
sensible defaults set, we will override several of these as an example
of how to do this. Config parameters not set in the dictionary will
automatically be set to their default values.

.. code:: ipython3

    gpz_train_dict = dict(n_basis=60, trainfrac=0.8, csl_method="normal", max_iter=150, hdf5_groupname="photometry") 

Let’s set up the training stage. We need to provide a name for the stage
for ceci, as well as a name for the model file that will be written by
the stage. We also include the arguments in the dictionary we wrote
above as additional arguments:

.. code:: ipython3

    # set up the stage to run our GPZ_training
    pz_train = GPzInformer.make_stage(name="GPz_Train", model="GPz_model.pkl", **gpz_train_dict)

We are now ready to run the stage to create the model. We will use the
training data from ``test_dc2_training_9816.hdf5``, which contains
10,225 galaxies drawn from healpix 9816 from the cosmoDC2_v1.1.4
dataset, to train the model. Note that we read this data in called
``train_data`` in the DataStore. Note that we set ``trainfrac`` to 0.8,
so 80% of the data will be used in the “main” training, but 20% will be
reserved by ``GPzInformer`` to determine a SIGMA parameter. We set
``max_iter`` to 150, so we will see 150 steps where the stage tries to
maximize the likelihood. We run the stage as follows:

.. code:: ipython3

    %%time
    pz_train.inform(training_data)


.. parsed-literal::

    Inserting handle into data store.  input: None, GPz_Train
    ngal: 10225
    training model...


.. parsed-literal::

    Iter	 logML/n 		 Train RMSE		 Train RMSE/n		 Valid RMSE		 Valid MLL		 Time    
       1	-3.4679027e-01	 3.2146656e-01	-3.3647052e-01	 3.1895781e-01	[-3.3036451e-01]	 4.5692515e-01


.. parsed-literal::

       2	-2.8037297e-01	 3.1295668e-01	-2.5804760e-01	 3.0865117e-01	[-2.4471887e-01]	 2.2519445e-01


.. parsed-literal::

       3	-2.3697965e-01	 2.9317674e-01	-1.9539562e-01	 2.8991659e-01	[-1.7959199e-01]	 2.7788520e-01


.. parsed-literal::

       4	-2.0458946e-01	 2.7664499e-01	-1.5747333e-01	 2.7480918e-01	[-1.3987143e-01]	 3.1469893e-01


.. parsed-literal::

       5	-1.4353447e-01	 2.6095582e-01	-1.0595711e-01	 2.5446943e-01	[-6.7744697e-02]	 2.0479417e-01


.. parsed-literal::

       6	-8.3680144e-02	 2.5500532e-01	-5.2797886e-02	 2.4658455e-01	[-1.8815065e-02]	 2.0376062e-01


.. parsed-literal::

       7	-6.2570211e-02	 2.5144542e-01	-3.8098851e-02	 2.4172780e-01	[-3.7523011e-06]	 2.0772886e-01
       8	-4.6929851e-02	 2.4863370e-01	-2.7655970e-02	 2.3913264e-01	[ 1.1723164e-02]	 1.8338108e-01


.. parsed-literal::

       9	-3.8910218e-02	 2.4730233e-01	-2.1374151e-02	 2.3811648e-01	[ 1.4009824e-02]	 2.0426130e-01


.. parsed-literal::

      10	-2.9983856e-02	 2.4559784e-01	-1.3888229e-02	 2.3686771e-01	[ 2.0228779e-02]	 2.0574737e-01
      11	-2.1321301e-02	 2.4418529e-01	-6.7805304e-03	 2.3435788e-01	[ 3.1853661e-02]	 1.6711164e-01


.. parsed-literal::

      12	-1.5541013e-02	 2.4307160e-01	-1.1825252e-03	 2.3409903e-01	[ 3.5973219e-02]	 2.0275116e-01
      13	-1.2213183e-02	 2.4243460e-01	 2.2138183e-03	 2.3332950e-01	[ 3.9879429e-02]	 1.7494202e-01


.. parsed-literal::

      14	 9.3975633e-02	 2.3010360e-01	 1.1586913e-01	 2.1962932e-01	[ 1.4976544e-01]	 3.1851339e-01


.. parsed-literal::

      15	 1.2000753e-01	 2.2561009e-01	 1.4230408e-01	 2.1769236e-01	[ 1.6262143e-01]	 3.2389069e-01


.. parsed-literal::

      16	 1.6930750e-01	 2.2059017e-01	 1.9224405e-01	 2.1411031e-01	[ 2.1152885e-01]	 2.0109057e-01
      17	 2.7380449e-01	 2.1343059e-01	 3.0343825e-01	 2.0437103e-01	[ 3.4852455e-01]	 1.8069077e-01


.. parsed-literal::

      18	 3.2188730e-01	 2.0816473e-01	 3.5640888e-01	 2.0155314e-01	[ 4.1479300e-01]	 1.9617414e-01
      19	 3.6026788e-01	 2.0627573e-01	 3.9329179e-01	 2.0031354e-01	[ 4.4758814e-01]	 1.8530130e-01


.. parsed-literal::

      20	 4.0881340e-01	 2.0167523e-01	 4.4223296e-01	 1.9645910e-01	[ 4.8583240e-01]	 2.0295572e-01
      21	 4.9099533e-01	 1.9521600e-01	 5.2478331e-01	 1.8641700e-01	[ 5.5460094e-01]	 1.9166732e-01


.. parsed-literal::

      22	 6.2528801e-01	 1.9267809e-01	 6.6379984e-01	 1.9010150e-01	[ 6.5060085e-01]	 2.0131493e-01
      23	 6.2691760e-01	 2.0175710e-01	 6.6911751e-01	 1.9895298e-01	  6.3813393e-01 	 1.8779325e-01


.. parsed-literal::

      24	 6.6874287e-01	 1.9382399e-01	 7.0747307e-01	 1.9234646e-01	[ 6.7999852e-01]	 1.8353343e-01


.. parsed-literal::

      25	 6.9992695e-01	 1.9262367e-01	 7.3753204e-01	 1.9122684e-01	[ 7.1574929e-01]	 2.0356321e-01
      26	 7.0688043e-01	 1.8960231e-01	 7.4401964e-01	 1.8989821e-01	[ 7.2286114e-01]	 1.9616079e-01


.. parsed-literal::

      27	 7.2659170e-01	 1.8657226e-01	 7.6440780e-01	 1.8823129e-01	[ 7.3652023e-01]	 2.1272421e-01
      28	 7.4505936e-01	 1.8515831e-01	 7.8387909e-01	 1.8755772e-01	[ 7.5518952e-01]	 1.8063879e-01


.. parsed-literal::

      29	 7.6485830e-01	 1.8375647e-01	 8.0375204e-01	 1.8771195e-01	[ 7.7335572e-01]	 2.0152092e-01
      30	 8.0322204e-01	 1.8316678e-01	 8.4310690e-01	 1.8877609e-01	[ 8.1554359e-01]	 2.0440292e-01


.. parsed-literal::

      31	 8.2355803e-01	 1.8515167e-01	 8.6385753e-01	 1.9300675e-01	[ 8.2998430e-01]	 2.0510483e-01


.. parsed-literal::

      32	 8.4413164e-01	 1.8157594e-01	 8.8498212e-01	 1.8814950e-01	[ 8.5418973e-01]	 2.1656442e-01
      33	 8.5932743e-01	 1.8021009e-01	 9.0091451e-01	 1.8724149e-01	[ 8.7161799e-01]	 2.0240641e-01


.. parsed-literal::

      34	 8.7630734e-01	 1.7887589e-01	 9.1872355e-01	 1.8816070e-01	[ 8.8836379e-01]	 1.7792821e-01
      35	 8.9981329e-01	 1.7712346e-01	 9.4325904e-01	 1.8586045e-01	[ 9.0837795e-01]	 1.9605899e-01


.. parsed-literal::

      36	 9.1234487e-01	 1.7939598e-01	 9.5582269e-01	 1.8750431e-01	[ 9.2349209e-01]	 1.9788861e-01
      37	 9.2577149e-01	 1.7759548e-01	 9.6887439e-01	 1.8550341e-01	[ 9.3509116e-01]	 1.8120623e-01


.. parsed-literal::

      38	 9.3703341e-01	 1.7740618e-01	 9.8018642e-01	 1.8436306e-01	[ 9.4622724e-01]	 1.8553185e-01
      39	 9.4956135e-01	 1.7720852e-01	 9.9306168e-01	 1.8302680e-01	[ 9.5752011e-01]	 1.9717598e-01


.. parsed-literal::

      40	 9.7189211e-01	 1.7587970e-01	 1.0166093e+00	 1.8051937e-01	[ 9.7374707e-01]	 2.1059060e-01


.. parsed-literal::

      41	 9.8099186e-01	 1.7456999e-01	 1.0262303e+00	 1.7836888e-01	[ 9.8148432e-01]	 3.1674504e-01


.. parsed-literal::

      42	 9.9135444e-01	 1.7308571e-01	 1.0368057e+00	 1.7748454e-01	[ 9.8534491e-01]	 2.1231365e-01


.. parsed-literal::

      43	 1.0060590e+00	 1.7101233e-01	 1.0522601e+00	 1.7636194e-01	[ 9.9138764e-01]	 2.1340919e-01
      44	 1.0168785e+00	 1.6990998e-01	 1.0632854e+00	 1.7512158e-01	[ 1.0043004e+00]	 1.8017554e-01


.. parsed-literal::

      45	 1.0306027e+00	 1.6837343e-01	 1.0775055e+00	 1.7294189e-01	[ 1.0190848e+00]	 2.0921493e-01


.. parsed-literal::

      46	 1.0384486e+00	 1.6724834e-01	 1.0856657e+00	 1.7107897e-01	[ 1.0336663e+00]	 3.1228995e-01
      47	 1.0472114e+00	 1.6533602e-01	 1.0940154e+00	 1.6950914e-01	[ 1.0393832e+00]	 1.9521189e-01


.. parsed-literal::

      48	 1.0549973e+00	 1.6397611e-01	 1.1021343e+00	 1.6768830e-01	[ 1.0451907e+00]	 2.1364737e-01
      49	 1.0608226e+00	 1.6296936e-01	 1.1082576e+00	 1.6654038e-01	[ 1.0520801e+00]	 1.7851305e-01


.. parsed-literal::

      50	 1.0667271e+00	 1.6214752e-01	 1.1144967e+00	 1.6571501e-01	[ 1.0572563e+00]	 1.9814897e-01
      51	 1.0772961e+00	 1.6120730e-01	 1.1253522e+00	 1.6544962e-01	[ 1.0618151e+00]	 2.0357800e-01


.. parsed-literal::

      52	 1.0842107e+00	 1.6143523e-01	 1.1324490e+00	 1.6619050e-01	[ 1.0666201e+00]	 1.7199969e-01


.. parsed-literal::

      53	 1.0929566e+00	 1.6103085e-01	 1.1407995e+00	 1.6603300e-01	[ 1.0710856e+00]	 2.0964050e-01
      54	 1.0975089e+00	 1.6081950e-01	 1.1451643e+00	 1.6555120e-01	[ 1.0741941e+00]	 1.7172861e-01


.. parsed-literal::

      55	 1.1030815e+00	 1.6066866e-01	 1.1507480e+00	 1.6541215e-01	[ 1.0773905e+00]	 1.7555475e-01


.. parsed-literal::

      56	 1.1084914e+00	 1.6097193e-01	 1.1567487e+00	 1.6580648e-01	[ 1.0789599e+00]	 2.0899868e-01
      57	 1.1141462e+00	 1.6048527e-01	 1.1625982e+00	 1.6536320e-01	[ 1.0834932e+00]	 1.8864679e-01


.. parsed-literal::

      58	 1.1200498e+00	 1.5990814e-01	 1.1690138e+00	 1.6488754e-01	[ 1.0883344e+00]	 1.9855285e-01


.. parsed-literal::

      59	 1.1247853e+00	 1.5952887e-01	 1.1737996e+00	 1.6453159e-01	[ 1.0946485e+00]	 2.0868111e-01


.. parsed-literal::

      60	 1.1321520e+00	 1.5879826e-01	 1.1813919e+00	 1.6397809e-01	[ 1.0990293e+00]	 2.1883750e-01


.. parsed-literal::

      61	 1.1364120e+00	 1.5828219e-01	 1.1855132e+00	 1.6363820e-01	[ 1.1077637e+00]	 2.0986533e-01
      62	 1.1418211e+00	 1.5749042e-01	 1.1907492e+00	 1.6229292e-01	[ 1.1125999e+00]	 1.9466972e-01


.. parsed-literal::

      63	 1.1474252e+00	 1.5659271e-01	 1.1965267e+00	 1.6076980e-01	[ 1.1156431e+00]	 2.0134282e-01


.. parsed-literal::

      64	 1.1510846e+00	 1.5583114e-01	 1.2004144e+00	 1.5986872e-01	[ 1.1176482e+00]	 2.0934916e-01


.. parsed-literal::

      65	 1.1574062e+00	 1.5411290e-01	 1.2072936e+00	 1.5803849e-01	[ 1.1220914e+00]	 2.0465589e-01
      66	 1.1604624e+00	 1.5248482e-01	 1.2107576e+00	 1.5717507e-01	[ 1.1230374e+00]	 2.0197034e-01


.. parsed-literal::

      67	 1.1656237e+00	 1.5239130e-01	 1.2156558e+00	 1.5729159e-01	[ 1.1283856e+00]	 1.9922042e-01


.. parsed-literal::

      68	 1.1683361e+00	 1.5215342e-01	 1.2183212e+00	 1.5733155e-01	[ 1.1307018e+00]	 2.1711731e-01


.. parsed-literal::

      69	 1.1737050e+00	 1.5175172e-01	 1.2237060e+00	 1.5753380e-01	[ 1.1341050e+00]	 2.0816541e-01
      70	 1.1778175e+00	 1.5142528e-01	 1.2280426e+00	 1.5777309e-01	[ 1.1367815e+00]	 1.7945051e-01


.. parsed-literal::

      71	 1.1826770e+00	 1.5103052e-01	 1.2328835e+00	 1.5734723e-01	[ 1.1425412e+00]	 2.0501709e-01


.. parsed-literal::

      72	 1.1858956e+00	 1.5085407e-01	 1.2362380e+00	 1.5704866e-01	[ 1.1468667e+00]	 2.1143126e-01
      73	 1.1895306e+00	 1.5071820e-01	 1.2401130e+00	 1.5706500e-01	[ 1.1494352e+00]	 1.7928123e-01


.. parsed-literal::

      74	 1.1937557e+00	 1.5060038e-01	 1.2447555e+00	 1.5706304e-01	[ 1.1556145e+00]	 1.7784810e-01


.. parsed-literal::

      75	 1.1982294e+00	 1.5009584e-01	 1.2496516e+00	 1.5744349e-01	  1.1553275e+00 	 2.1208787e-01
      76	 1.2014261e+00	 1.4984129e-01	 1.2526694e+00	 1.5735231e-01	[ 1.1584919e+00]	 1.7107749e-01


.. parsed-literal::

      77	 1.2068875e+00	 1.4920399e-01	 1.2580504e+00	 1.5742961e-01	[ 1.1638710e+00]	 2.1053648e-01


.. parsed-literal::

      78	 1.2100115e+00	 1.4819028e-01	 1.2615430e+00	 1.5607269e-01	[ 1.1715639e+00]	 2.0618510e-01
      79	 1.2139287e+00	 1.4793975e-01	 1.2654685e+00	 1.5606062e-01	[ 1.1736522e+00]	 1.8620038e-01


.. parsed-literal::

      80	 1.2169736e+00	 1.4753877e-01	 1.2686962e+00	 1.5583452e-01	[ 1.1758811e+00]	 2.0430994e-01


.. parsed-literal::

      81	 1.2195128e+00	 1.4710154e-01	 1.2714553e+00	 1.5551081e-01	[ 1.1762948e+00]	 2.1631813e-01


.. parsed-literal::

      82	 1.2227287e+00	 1.4646935e-01	 1.2747816e+00	 1.5486963e-01	[ 1.1785357e+00]	 2.1213341e-01
      83	 1.2254712e+00	 1.4590922e-01	 1.2775994e+00	 1.5451825e-01	[ 1.1807455e+00]	 1.9698119e-01


.. parsed-literal::

      84	 1.2299287e+00	 1.4517327e-01	 1.2820793e+00	 1.5386558e-01	[ 1.1835866e+00]	 1.6910982e-01


.. parsed-literal::

      85	 1.2340042e+00	 1.4460924e-01	 1.2861884e+00	 1.5375625e-01	[ 1.1896702e+00]	 2.0733285e-01
      86	 1.2373067e+00	 1.4441412e-01	 1.2893921e+00	 1.5326464e-01	[ 1.1924298e+00]	 1.9756818e-01


.. parsed-literal::

      87	 1.2408835e+00	 1.4420713e-01	 1.2929676e+00	 1.5281101e-01	[ 1.1954038e+00]	 1.7687750e-01
      88	 1.2436688e+00	 1.4417343e-01	 1.2957680e+00	 1.5255378e-01	[ 1.1982425e+00]	 1.7101312e-01


.. parsed-literal::

      89	 1.2477429e+00	 1.4413823e-01	 1.2999503e+00	 1.5233577e-01	[ 1.2024214e+00]	 2.0500374e-01


.. parsed-literal::

      90	 1.2521347e+00	 1.4411774e-01	 1.3044519e+00	 1.5232107e-01	[ 1.2074868e+00]	 2.1655774e-01
      91	 1.2566028e+00	 1.4396189e-01	 1.3089288e+00	 1.5206265e-01	[ 1.2124780e+00]	 1.8251300e-01


.. parsed-literal::

      92	 1.2600861e+00	 1.4356764e-01	 1.3123790e+00	 1.5162205e-01	[ 1.2154639e+00]	 1.9819832e-01
      93	 1.2639549e+00	 1.4310299e-01	 1.3161662e+00	 1.5108414e-01	[ 1.2186083e+00]	 1.7939687e-01


.. parsed-literal::

      94	 1.2672652e+00	 1.4254662e-01	 1.3196580e+00	 1.5078239e-01	  1.2170546e+00 	 1.6704988e-01


.. parsed-literal::

      95	 1.2704397e+00	 1.4208468e-01	 1.3228676e+00	 1.4985848e-01	  1.2182985e+00 	 2.1446991e-01
      96	 1.2729119e+00	 1.4204006e-01	 1.3253790e+00	 1.4960211e-01	[ 1.2197776e+00]	 1.8391943e-01


.. parsed-literal::

      97	 1.2766789e+00	 1.4193252e-01	 1.3293282e+00	 1.4905175e-01	[ 1.2211657e+00]	 2.0020771e-01
      98	 1.2816566e+00	 1.4172477e-01	 1.3344474e+00	 1.4846514e-01	[ 1.2243281e+00]	 1.7708898e-01


.. parsed-literal::

      99	 1.2857722e+00	 1.4161271e-01	 1.3387237e+00	 1.4781094e-01	[ 1.2265564e+00]	 2.0457387e-01


.. parsed-literal::

     100	 1.2884912e+00	 1.4149943e-01	 1.3414546e+00	 1.4771586e-01	[ 1.2287281e+00]	 2.0868587e-01
     101	 1.2915561e+00	 1.4136905e-01	 1.3445598e+00	 1.4767804e-01	[ 1.2288892e+00]	 1.9087458e-01


.. parsed-literal::

     102	 1.2948310e+00	 1.4148223e-01	 1.3479159e+00	 1.4775013e-01	[ 1.2312036e+00]	 1.9428015e-01
     103	 1.2991644e+00	 1.4169506e-01	 1.3524953e+00	 1.4767405e-01	[ 1.2314221e+00]	 1.9268441e-01


.. parsed-literal::

     104	 1.3023180e+00	 1.4199676e-01	 1.3558600e+00	 1.4774979e-01	[ 1.2323700e+00]	 2.0449305e-01
     105	 1.3054563e+00	 1.4201581e-01	 1.3590424e+00	 1.4757106e-01	[ 1.2356016e+00]	 1.7113757e-01


.. parsed-literal::

     106	 1.3076336e+00	 1.4188931e-01	 1.3611933e+00	 1.4735691e-01	[ 1.2380559e+00]	 2.0334625e-01
     107	 1.3118188e+00	 1.4164595e-01	 1.3653463e+00	 1.4703416e-01	[ 1.2401973e+00]	 1.7817283e-01


.. parsed-literal::

     108	 1.3141093e+00	 1.4118676e-01	 1.3680462e+00	 1.4624311e-01	  1.2368009e+00 	 1.7472363e-01
     109	 1.3191336e+00	 1.4101830e-01	 1.3729270e+00	 1.4626816e-01	  1.2362905e+00 	 2.0326757e-01


.. parsed-literal::

     110	 1.3211354e+00	 1.4097805e-01	 1.3749690e+00	 1.4621047e-01	  1.2358532e+00 	 2.0428777e-01
     111	 1.3244187e+00	 1.4090291e-01	 1.3785156e+00	 1.4604612e-01	  1.2327625e+00 	 2.0610619e-01


.. parsed-literal::

     112	 1.3269800e+00	 1.4090437e-01	 1.3814118e+00	 1.4552777e-01	  1.2336228e+00 	 2.0830345e-01


.. parsed-literal::

     113	 1.3301704e+00	 1.4085994e-01	 1.3846686e+00	 1.4554441e-01	  1.2338102e+00 	 2.1104574e-01


.. parsed-literal::

     114	 1.3328030e+00	 1.4084917e-01	 1.3873499e+00	 1.4550724e-01	  1.2342990e+00 	 2.0901656e-01


.. parsed-literal::

     115	 1.3351011e+00	 1.4082496e-01	 1.3896542e+00	 1.4538916e-01	  1.2335187e+00 	 2.1100140e-01


.. parsed-literal::

     116	 1.3380573e+00	 1.4067878e-01	 1.3927507e+00	 1.4489968e-01	  1.2269658e+00 	 2.1866584e-01


.. parsed-literal::

     117	 1.3412858e+00	 1.4068271e-01	 1.3959359e+00	 1.4471690e-01	  1.2273114e+00 	 2.0697951e-01


.. parsed-literal::

     118	 1.3429827e+00	 1.4061509e-01	 1.3975882e+00	 1.4456133e-01	  1.2292726e+00 	 2.0738411e-01
     119	 1.3453647e+00	 1.4046061e-01	 1.4000555e+00	 1.4413146e-01	  1.2307038e+00 	 1.9712043e-01


.. parsed-literal::

     120	 1.3469932e+00	 1.4055826e-01	 1.4019345e+00	 1.4396226e-01	  1.2353284e+00 	 1.9921494e-01
     121	 1.3495717e+00	 1.4030578e-01	 1.4045017e+00	 1.4351784e-01	  1.2357196e+00 	 2.0303440e-01


.. parsed-literal::

     122	 1.3516721e+00	 1.4012660e-01	 1.4066867e+00	 1.4309413e-01	  1.2355744e+00 	 1.9955182e-01
     123	 1.3533410e+00	 1.4004171e-01	 1.4084072e+00	 1.4284269e-01	  1.2358029e+00 	 1.8369174e-01


.. parsed-literal::

     124	 1.3566178e+00	 1.3969814e-01	 1.4118101e+00	 1.4210402e-01	  1.2365890e+00 	 1.9579768e-01
     125	 1.3580343e+00	 1.3961942e-01	 1.4134391e+00	 1.4143992e-01	  1.2367004e+00 	 1.9739032e-01


.. parsed-literal::

     126	 1.3609780e+00	 1.3952479e-01	 1.4162203e+00	 1.4158610e-01	  1.2390416e+00 	 2.1005464e-01


.. parsed-literal::

     127	 1.3626403e+00	 1.3945676e-01	 1.4178368e+00	 1.4161138e-01	[ 1.2402673e+00]	 2.1608639e-01


.. parsed-literal::

     128	 1.3650796e+00	 1.3937019e-01	 1.4202960e+00	 1.4158992e-01	[ 1.2414521e+00]	 2.1883798e-01
     129	 1.3682376e+00	 1.3927674e-01	 1.4236527e+00	 1.4142177e-01	  1.2369341e+00 	 1.7910028e-01


.. parsed-literal::

     130	 1.3703025e+00	 1.3943591e-01	 1.4259631e+00	 1.4156359e-01	  1.2365339e+00 	 2.0111728e-01
     131	 1.3723985e+00	 1.3934435e-01	 1.4279728e+00	 1.4126993e-01	  1.2376107e+00 	 1.9125295e-01


.. parsed-literal::

     132	 1.3739337e+00	 1.3930518e-01	 1.4295184e+00	 1.4099167e-01	  1.2367183e+00 	 2.0744109e-01


.. parsed-literal::

     133	 1.3760272e+00	 1.3935708e-01	 1.4316699e+00	 1.4080383e-01	  1.2345559e+00 	 2.1080232e-01


.. parsed-literal::

     134	 1.3776732e+00	 1.3917592e-01	 1.4333775e+00	 1.4025322e-01	  1.2286518e+00 	 3.2887125e-01


.. parsed-literal::

     135	 1.3799197e+00	 1.3932738e-01	 1.4357132e+00	 1.4033901e-01	  1.2235819e+00 	 2.1694350e-01


.. parsed-literal::

     136	 1.3814755e+00	 1.3933720e-01	 1.4372934e+00	 1.4033865e-01	  1.2218369e+00 	 2.1136856e-01


.. parsed-literal::

     137	 1.3835323e+00	 1.3932884e-01	 1.4394016e+00	 1.4038121e-01	  1.2184754e+00 	 2.1682405e-01
     138	 1.3859574e+00	 1.3917254e-01	 1.4419550e+00	 1.3999023e-01	  1.2145901e+00 	 1.9782519e-01


.. parsed-literal::

     139	 1.3884966e+00	 1.3916134e-01	 1.4446091e+00	 1.4014946e-01	  1.2014750e+00 	 1.9965458e-01


.. parsed-literal::

     140	 1.3905140e+00	 1.3900461e-01	 1.4466662e+00	 1.3998514e-01	  1.1982342e+00 	 2.0193791e-01
     141	 1.3930149e+00	 1.3871085e-01	 1.4493048e+00	 1.3955263e-01	  1.1880429e+00 	 1.8654728e-01


.. parsed-literal::

     142	 1.3949862e+00	 1.3848594e-01	 1.4514173e+00	 1.3938811e-01	  1.1752196e+00 	 1.9845462e-01
     143	 1.3975678e+00	 1.3818479e-01	 1.4541899e+00	 1.3905378e-01	  1.1593854e+00 	 1.8762422e-01


.. parsed-literal::

     144	 1.3993391e+00	 1.3802428e-01	 1.4562034e+00	 1.3897139e-01	  1.1332099e+00 	 2.1023703e-01


.. parsed-literal::

     145	 1.4014212e+00	 1.3768606e-01	 1.4581759e+00	 1.3858195e-01	  1.1422644e+00 	 2.0744228e-01


.. parsed-literal::

     146	 1.4028730e+00	 1.3756357e-01	 1.4595135e+00	 1.3849614e-01	  1.1534131e+00 	 2.0423770e-01
     147	 1.4052805e+00	 1.3724432e-01	 1.4619161e+00	 1.3818889e-01	  1.1578863e+00 	 1.8539476e-01


.. parsed-literal::

     148	 1.4061944e+00	 1.3724338e-01	 1.4630042e+00	 1.3833451e-01	  1.1696988e+00 	 1.9927621e-01


.. parsed-literal::

     149	 1.4086763e+00	 1.3700834e-01	 1.4654026e+00	 1.3802211e-01	  1.1616510e+00 	 2.1359921e-01


.. parsed-literal::

     150	 1.4098944e+00	 1.3694740e-01	 1.4666394e+00	 1.3795192e-01	  1.1545000e+00 	 2.0434213e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 2s, sys: 1.15 s, total: 2min 3s
    Wall time: 31 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f1978cc4eb0>



This should have taken about 30 seconds on a typical desktop computer,
and you should now see a file called ``GPz_model.pkl`` in the directory.
This model file is used by the ``GPzEstimator`` stage to determine our
redshift PDFs for the test set of galaxies. Let’s set up that stage,
again defining a dictionary of variables for the config params:

.. code:: ipython3

    gpz_test_dict = dict(hdf5_groupname="photometry", model="GPz_model.pkl")
    
    gpz_run = GPzEstimator.make_stage(name="gpz_run", **gpz_test_dict)

Let’s run the stage and compute photo-z’s for our test set:

.. code:: ipython3

    %%time
    results = gpz_run.estimate(test_data)


.. parsed-literal::

    Inserting handle into data store.  input: None, gpz_run
    Inserting handle into data store.  model: GPz_model.pkl, gpz_run
    Process 0 running estimator on chunk 0 - 20,449
    Process 0 estimating GPz PZ PDF for rows 0 - 20,449


.. parsed-literal::

    Inserting handle into data store.  output_gpz_run: inprogress_output_gpz_run.hdf5, gpz_run
    CPU times: user 966 ms, sys: 46 ms, total: 1.01 s
    Wall time: 372 ms


This should be very fast, under a second for our 20,449 galaxies in the
test set. Now, let’s plot a scatter plot of the point estimates, as well
as a few example PDFs. We can get access to the ``qp`` ensemble that was
written via the DataStore via ``results()``

.. code:: ipython3

    ens = results()

.. code:: ipython3

    expdfids = [2, 180, 13517, 18032]
    fig, axs = plt.subplots(4, 1, figsize=(12,10))
    for i, xx in enumerate(expdfids):
        axs[i].set_xlim(0,3)
        ens[xx].plot_native(axes=axs[i])
    axs[3].set_xlabel("redshift", fontsize=15)




.. parsed-literal::

    Text(0.5, 0, 'redshift')




.. image:: 06_GPz_files/06_GPz_15_1.png


GPzEstimator parameterizes each PDF as a single Gaussian, here we see a
few examples of Gaussians of different widths. Now let’s grab the mode
of each PDF (stored as ancil data in the ensemble) and compare to the
true redshifts from the test_data file:

.. code:: ipython3

    truez = test_data['photometry']['redshift']
    zmode = ens.ancil['zmode'].flatten()

.. code:: ipython3

    plt.figure(figsize=(12,12))
    plt.scatter(truez, zmode, s=3)
    plt.plot([0,3],[0,3], 'k--')
    plt.xlabel("redshift", fontsize=12)
    plt.ylabel("z mode", fontsize=12)




.. parsed-literal::

    Text(0, 0.5, 'z mode')




.. image:: 06_GPz_files/06_GPz_18_1.png

