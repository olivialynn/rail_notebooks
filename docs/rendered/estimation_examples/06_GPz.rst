GPz Estimation Example
======================

**Author:** Sam Schmidt

**Last Run Successfully:** September 26, 2023

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
    import qp
    from rail.core.data import TableHandle
    from rail.core.stage import RailStage
    from rail.estimation.algos.gpz import GPzInformer, GPzEstimator

.. code:: ipython3

    # set up the DataStore to keep track of data
    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True

.. code:: ipython3

    # find_rail_file is a convenience function that finds a file in the RAIL ecosystem   We have several example data files that are copied with RAIL that we can use for our example run, let's grab those files, one for training/validation, and the other for testing:
    from rail.utils.path_utils import find_rail_file
    trainFile = find_rail_file('examples_data/testdata/test_dc2_training_9816.hdf5')
    testFile = find_rail_file('examples_data/testdata/test_dc2_validation_9816.hdf5')
    training_data = DS.read_file("training_data", TableHandle, trainFile)
    test_data = DS.read_file("test_data", TableHandle, testFile)

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

    ngal: 10225
    training model...


.. parsed-literal::

    Iter	 logML/n 		 Train RMSE		 Train RMSE/n		 Valid RMSE		 Valid MLL		 Time    
       1	-3.5340872e-01	 3.2368291e-01	-3.4368818e-01	 3.0865032e-01	[-3.1611143e-01]	 4.6135783e-01


.. parsed-literal::

       2	-2.8025332e-01	 3.1200850e-01	-2.5525321e-01	 2.9753295e-01	[-2.1296775e-01]	 2.3416328e-01


.. parsed-literal::

       3	-2.3539678e-01	 2.9117902e-01	-1.9204878e-01	 2.8012979e-01	[-1.4747082e-01]	 2.8315473e-01
       4	-2.0751949e-01	 2.6762518e-01	-1.6842067e-01	 2.5656133e-01	[-1.0546022e-01]	 1.7094636e-01


.. parsed-literal::

       5	-1.1095401e-01	 2.5880732e-01	-7.4540180e-02	 2.5070109e-01	[-3.6294944e-02]	 1.8675923e-01


.. parsed-literal::

       6	-7.8867359e-02	 2.5378706e-01	-4.7523205e-02	 2.4511195e-01	[-1.4271067e-02]	 2.1306705e-01


.. parsed-literal::

       7	-5.8221752e-02	 2.5037679e-01	-3.4326241e-02	 2.4101976e-01	[ 1.4223917e-03]	 2.1151996e-01
       8	-4.7964641e-02	 2.4871806e-01	-2.7463296e-02	 2.3957678e-01	[ 7.8722443e-03]	 1.8411803e-01


.. parsed-literal::

       9	-3.4999158e-02	 2.4634214e-01	-1.7443310e-02	 2.3764160e-01	[ 1.6522190e-02]	 2.0683312e-01


.. parsed-literal::

      10	-2.3837381e-02	 2.4415746e-01	-8.2416515e-03	 2.3522054e-01	[ 2.8583565e-02]	 2.1269798e-01


.. parsed-literal::

      11	-2.0064872e-02	 2.4366939e-01	-5.8027660e-03	 2.3552217e-01	  2.5852275e-02 	 2.1487617e-01
      12	-1.5956639e-02	 2.4289315e-01	-1.9541592e-03	 2.3455539e-01	[ 3.2398656e-02]	 1.9314170e-01


.. parsed-literal::

      13	-1.2713347e-02	 2.4218460e-01	 1.3064497e-03	 2.3372734e-01	[ 3.6048914e-02]	 2.0904088e-01


.. parsed-literal::

      14	-8.9931510e-03	 2.4136422e-01	 5.4005241e-03	 2.3313134e-01	[ 4.0667871e-02]	 2.0731473e-01


.. parsed-literal::

      15	 4.9489446e-02	 2.2265951e-01	 7.1778853e-02	 2.1760249e-01	[ 7.8049979e-02]	 4.1420841e-01
      16	 2.4900737e-01	 2.1444870e-01	 2.7604420e-01	 2.1055739e-01	[ 2.8087381e-01]	 1.9079328e-01


.. parsed-literal::

      17	 2.9588251e-01	 2.0989812e-01	 3.2610659e-01	 2.0722959e-01	[ 3.2818103e-01]	 3.0736971e-01


.. parsed-literal::

      18	 3.2087146e-01	 2.0766646e-01	 3.5172214e-01	 2.0507960e-01	[ 3.5170282e-01]	 2.1258163e-01


.. parsed-literal::

      19	 3.5214027e-01	 2.0618052e-01	 3.8309614e-01	 2.0457022e-01	[ 3.7681367e-01]	 2.1337152e-01


.. parsed-literal::

      20	 3.9896542e-01	 2.0316172e-01	 4.3025068e-01	 2.0105979e-01	[ 4.2243798e-01]	 2.0593739e-01
      21	 5.2436948e-01	 2.0239485e-01	 5.5811212e-01	 1.9653621e-01	[ 5.4116567e-01]	 1.7005825e-01


.. parsed-literal::

      22	 5.9312013e-01	 1.9768995e-01	 6.3061535e-01	 1.9224090e-01	[ 6.0035928e-01]	 3.1712055e-01
      23	 6.3382248e-01	 1.9638698e-01	 6.7196119e-01	 1.8963273e-01	[ 6.3502990e-01]	 1.9550896e-01


.. parsed-literal::

      24	 6.5744659e-01	 1.9352866e-01	 6.9378063e-01	 1.8729454e-01	[ 6.6469475e-01]	 2.1151471e-01


.. parsed-literal::

      25	 6.7513169e-01	 1.9201010e-01	 7.1156000e-01	 1.8549128e-01	[ 6.8240656e-01]	 2.0620990e-01


.. parsed-literal::

      26	 6.9042273e-01	 1.9061358e-01	 7.2656224e-01	 1.8316811e-01	[ 6.9864385e-01]	 3.3275342e-01


.. parsed-literal::

      27	 7.0458684e-01	 1.9037581e-01	 7.4143761e-01	 1.8224285e-01	[ 7.1309428e-01]	 2.1381426e-01
      28	 7.2095140e-01	 1.8876582e-01	 7.5817665e-01	 1.8132611e-01	[ 7.2729626e-01]	 1.8902206e-01


.. parsed-literal::

      29	 7.4194916e-01	 1.9012026e-01	 7.7911618e-01	 1.8381237e-01	[ 7.4630464e-01]	 2.0979095e-01


.. parsed-literal::

      30	 7.6825688e-01	 1.8855382e-01	 8.0551564e-01	 1.8357643e-01	[ 7.7057575e-01]	 2.1068358e-01
      31	 7.9383705e-01	 1.9007540e-01	 8.3156381e-01	 1.8436995e-01	[ 8.0515730e-01]	 2.0012450e-01


.. parsed-literal::

      32	 8.1982860e-01	 1.8929418e-01	 8.5833310e-01	 1.8508935e-01	[ 8.3327294e-01]	 2.0977283e-01


.. parsed-literal::

      33	 8.4629646e-01	 1.8866547e-01	 8.8531106e-01	 1.8558047e-01	[ 8.5500089e-01]	 2.0569301e-01
      34	 8.5720291e-01	 1.8914143e-01	 8.9718425e-01	 1.8765773e-01	[ 8.6141401e-01]	 1.9984436e-01


.. parsed-literal::

      35	 8.7249752e-01	 1.8650435e-01	 9.1212781e-01	 1.8441446e-01	[ 8.7193919e-01]	 2.0568419e-01
      36	 8.8366817e-01	 1.8294512e-01	 9.2308678e-01	 1.8205599e-01	[ 8.7879352e-01]	 1.7708206e-01


.. parsed-literal::

      37	 8.9977761e-01	 1.8020071e-01	 9.3931405e-01	 1.8034109e-01	[ 8.9407477e-01]	 2.1398973e-01
      38	 9.1929882e-01	 1.7921716e-01	 9.6001479e-01	 1.7938444e-01	[ 9.1852829e-01]	 1.9961071e-01


.. parsed-literal::

      39	 9.3352123e-01	 1.7979024e-01	 9.7477330e-01	 1.7893599e-01	[ 9.3222680e-01]	 2.0234251e-01


.. parsed-literal::

      40	 9.4350943e-01	 1.7965653e-01	 9.8475278e-01	 1.7842376e-01	[ 9.4319378e-01]	 2.2233295e-01


.. parsed-literal::

      41	 9.6117096e-01	 1.7954480e-01	 1.0030361e+00	 1.7809098e-01	[ 9.6572348e-01]	 2.0351434e-01


.. parsed-literal::

      42	 9.6911641e-01	 1.7964505e-01	 1.0113550e+00	 1.7803726e-01	[ 9.7687699e-01]	 2.1355534e-01
      43	 9.7770143e-01	 1.7889313e-01	 1.0201351e+00	 1.7692111e-01	[ 9.8384747e-01]	 1.7164588e-01


.. parsed-literal::

      44	 9.8837296e-01	 1.7673081e-01	 1.0313073e+00	 1.7437923e-01	[ 9.8982343e-01]	 2.0381165e-01
      45	 9.9746396e-01	 1.7681219e-01	 1.0407297e+00	 1.7357206e-01	[ 9.9923063e-01]	 1.9076490e-01


.. parsed-literal::

      46	 1.0108602e+00	 1.7564071e-01	 1.0545806e+00	 1.7235428e-01	[ 1.0116767e+00]	 1.7021728e-01
      47	 1.0207237e+00	 1.7475713e-01	 1.0645297e+00	 1.7188042e-01	[ 1.0222703e+00]	 2.0288563e-01


.. parsed-literal::

      48	 1.0298838e+00	 1.7459847e-01	 1.0740111e+00	 1.7241387e-01	[ 1.0358471e+00]	 1.8932056e-01


.. parsed-literal::

      49	 1.0375051e+00	 1.7311633e-01	 1.0815996e+00	 1.7099627e-01	[ 1.0439505e+00]	 2.1657848e-01


.. parsed-literal::

      50	 1.0449480e+00	 1.7173262e-01	 1.0890546e+00	 1.6967331e-01	[ 1.0527825e+00]	 2.1076941e-01
      51	 1.0539140e+00	 1.6968193e-01	 1.0986354e+00	 1.6792293e-01	[ 1.0617400e+00]	 1.7825937e-01


.. parsed-literal::

      52	 1.0637392e+00	 1.6826606e-01	 1.1089174e+00	 1.6625290e-01	[ 1.0750028e+00]	 2.1666050e-01
      53	 1.0705237e+00	 1.6776363e-01	 1.1157061e+00	 1.6575445e-01	[ 1.0818253e+00]	 1.9846749e-01


.. parsed-literal::

      54	 1.0783240e+00	 1.6649857e-01	 1.1238487e+00	 1.6498007e-01	[ 1.0886986e+00]	 2.1303773e-01
      55	 1.0857600e+00	 1.6543208e-01	 1.1313358e+00	 1.6464327e-01	[ 1.0891560e+00]	 1.8690062e-01


.. parsed-literal::

      56	 1.0923512e+00	 1.6351383e-01	 1.1379342e+00	 1.6368198e-01	[ 1.0923222e+00]	 2.0307970e-01


.. parsed-literal::

      57	 1.0984866e+00	 1.6202633e-01	 1.1442649e+00	 1.6303532e-01	[ 1.0976212e+00]	 2.1561265e-01
      58	 1.1041843e+00	 1.6123378e-01	 1.1501198e+00	 1.6273950e-01	[ 1.1029364e+00]	 1.9781590e-01


.. parsed-literal::

      59	 1.1103244e+00	 1.5931560e-01	 1.1567279e+00	 1.6202691e-01	[ 1.1062194e+00]	 2.0052266e-01


.. parsed-literal::

      60	 1.1169857e+00	 1.5837600e-01	 1.1634038e+00	 1.6138541e-01	[ 1.1100466e+00]	 2.1580315e-01


.. parsed-literal::

      61	 1.1253325e+00	 1.5701099e-01	 1.1717624e+00	 1.6042649e-01	[ 1.1125779e+00]	 2.2031641e-01
      62	 1.1299582e+00	 1.5562554e-01	 1.1765932e+00	 1.5976977e-01	  1.1104281e+00 	 1.9727254e-01


.. parsed-literal::

      63	 1.1360397e+00	 1.5566118e-01	 1.1824123e+00	 1.5954045e-01	[ 1.1190878e+00]	 1.7353487e-01


.. parsed-literal::

      64	 1.1409394e+00	 1.5527201e-01	 1.1874505e+00	 1.5956161e-01	[ 1.1237729e+00]	 2.0788026e-01
      65	 1.1460839e+00	 1.5489254e-01	 1.1927910e+00	 1.5946551e-01	[ 1.1285677e+00]	 1.8684769e-01


.. parsed-literal::

      66	 1.1527364e+00	 1.5421904e-01	 1.2000495e+00	 1.6023613e-01	[ 1.1288812e+00]	 2.0397401e-01


.. parsed-literal::

      67	 1.1602439e+00	 1.5398045e-01	 1.2076096e+00	 1.5996377e-01	[ 1.1340722e+00]	 2.1346378e-01


.. parsed-literal::

      68	 1.1650130e+00	 1.5368694e-01	 1.2123800e+00	 1.5993848e-01	[ 1.1370371e+00]	 2.1312213e-01


.. parsed-literal::

      69	 1.1732138e+00	 1.5317927e-01	 1.2206782e+00	 1.5965714e-01	[ 1.1405123e+00]	 2.1123719e-01


.. parsed-literal::

      70	 1.1775685e+00	 1.5346636e-01	 1.2252384e+00	 1.5988602e-01	[ 1.1464216e+00]	 3.1967878e-01


.. parsed-literal::

      71	 1.1825280e+00	 1.5267751e-01	 1.2302466e+00	 1.5900879e-01	[ 1.1508140e+00]	 2.0943189e-01


.. parsed-literal::

      72	 1.1887039e+00	 1.5202496e-01	 1.2367263e+00	 1.5818326e-01	[ 1.1560749e+00]	 2.0175242e-01


.. parsed-literal::

      73	 1.1925166e+00	 1.5061333e-01	 1.2406870e+00	 1.5664178e-01	[ 1.1600531e+00]	 2.0703292e-01
      74	 1.1968430e+00	 1.5042068e-01	 1.2451276e+00	 1.5646922e-01	[ 1.1638178e+00]	 1.6881323e-01


.. parsed-literal::

      75	 1.2016125e+00	 1.4960849e-01	 1.2500199e+00	 1.5597334e-01	[ 1.1656269e+00]	 2.0683908e-01


.. parsed-literal::

      76	 1.2060045e+00	 1.4878856e-01	 1.2545917e+00	 1.5546308e-01	[ 1.1679204e+00]	 2.0909095e-01
      77	 1.2109312e+00	 1.4718258e-01	 1.2598242e+00	 1.5450420e-01	  1.1652264e+00 	 1.7363811e-01


.. parsed-literal::

      78	 1.2164961e+00	 1.4662596e-01	 1.2653836e+00	 1.5377150e-01	[ 1.1735513e+00]	 1.9506621e-01


.. parsed-literal::

      79	 1.2206627e+00	 1.4626669e-01	 1.2694979e+00	 1.5324354e-01	[ 1.1788044e+00]	 2.1165204e-01


.. parsed-literal::

      80	 1.2251499e+00	 1.4591395e-01	 1.2740944e+00	 1.5283902e-01	[ 1.1827229e+00]	 2.1105313e-01


.. parsed-literal::

      81	 1.2285054e+00	 1.4577509e-01	 1.2776261e+00	 1.5318105e-01	  1.1781028e+00 	 2.1286154e-01


.. parsed-literal::

      82	 1.2343980e+00	 1.4545424e-01	 1.2835107e+00	 1.5304231e-01	[ 1.1839318e+00]	 2.1440840e-01
      83	 1.2377361e+00	 1.4521567e-01	 1.2868410e+00	 1.5303925e-01	[ 1.1860946e+00]	 1.8486166e-01


.. parsed-literal::

      84	 1.2420458e+00	 1.4487681e-01	 1.2912387e+00	 1.5312295e-01	  1.1860901e+00 	 2.0641947e-01
      85	 1.2448409e+00	 1.4472010e-01	 1.2942711e+00	 1.5326123e-01	  1.1833870e+00 	 1.9365478e-01


.. parsed-literal::

      86	 1.2499976e+00	 1.4423225e-01	 1.2994672e+00	 1.5306277e-01	  1.1846002e+00 	 2.0199895e-01


.. parsed-literal::

      87	 1.2526949e+00	 1.4401070e-01	 1.3021722e+00	 1.5284194e-01	[ 1.1868082e+00]	 2.1734047e-01
      88	 1.2552680e+00	 1.4369943e-01	 1.3048649e+00	 1.5249708e-01	[ 1.1876864e+00]	 1.9039893e-01


.. parsed-literal::

      89	 1.2599717e+00	 1.4317033e-01	 1.3097043e+00	 1.5190546e-01	[ 1.1924706e+00]	 2.2038007e-01
      90	 1.2641588e+00	 1.4210898e-01	 1.3142620e+00	 1.5088922e-01	[ 1.1932098e+00]	 1.9985962e-01


.. parsed-literal::

      91	 1.2702035e+00	 1.4185773e-01	 1.3202380e+00	 1.5073953e-01	[ 1.2001255e+00]	 1.9393945e-01


.. parsed-literal::

      92	 1.2742908e+00	 1.4164039e-01	 1.3243287e+00	 1.5080770e-01	[ 1.2044209e+00]	 2.1024871e-01


.. parsed-literal::

      93	 1.2789177e+00	 1.4120379e-01	 1.3291117e+00	 1.5074466e-01	[ 1.2084993e+00]	 2.1099567e-01


.. parsed-literal::

      94	 1.2820789e+00	 1.4137996e-01	 1.3325807e+00	 1.5132598e-01	[ 1.2113044e+00]	 2.1896911e-01


.. parsed-literal::

      95	 1.2877701e+00	 1.4057206e-01	 1.3382726e+00	 1.5072156e-01	[ 1.2146263e+00]	 2.1875238e-01
      96	 1.2913069e+00	 1.4016232e-01	 1.3418059e+00	 1.5022107e-01	[ 1.2176125e+00]	 1.8525600e-01


.. parsed-literal::

      97	 1.2955766e+00	 1.3980651e-01	 1.3461791e+00	 1.4974312e-01	[ 1.2205911e+00]	 1.8695092e-01
      98	 1.2999065e+00	 1.3949720e-01	 1.3506594e+00	 1.4912339e-01	[ 1.2219862e+00]	 2.0277882e-01


.. parsed-literal::

      99	 1.3041294e+00	 1.3950371e-01	 1.3548969e+00	 1.4908375e-01	[ 1.2256369e+00]	 2.1311378e-01


.. parsed-literal::

     100	 1.3070036e+00	 1.3955609e-01	 1.3578021e+00	 1.4914905e-01	[ 1.2276176e+00]	 2.0758367e-01


.. parsed-literal::

     101	 1.3103885e+00	 1.3951857e-01	 1.3612707e+00	 1.4907297e-01	[ 1.2294734e+00]	 2.0642781e-01


.. parsed-literal::

     102	 1.3142649e+00	 1.3969365e-01	 1.3654343e+00	 1.4910195e-01	  1.2266312e+00 	 2.2150302e-01


.. parsed-literal::

     103	 1.3187097e+00	 1.3928269e-01	 1.3699177e+00	 1.4880255e-01	[ 1.2305343e+00]	 2.1880960e-01
     104	 1.3213297e+00	 1.3895677e-01	 1.3725024e+00	 1.4862438e-01	[ 1.2315542e+00]	 1.9959140e-01


.. parsed-literal::

     105	 1.3247688e+00	 1.3848240e-01	 1.3760527e+00	 1.4841946e-01	  1.2314564e+00 	 1.8698549e-01


.. parsed-literal::

     106	 1.3285731e+00	 1.3799475e-01	 1.3800915e+00	 1.4838359e-01	  1.2261097e+00 	 2.0916820e-01


.. parsed-literal::

     107	 1.3321136e+00	 1.3773328e-01	 1.3837810e+00	 1.4830198e-01	  1.2275682e+00 	 2.0142293e-01


.. parsed-literal::

     108	 1.3346634e+00	 1.3759986e-01	 1.3863442e+00	 1.4838120e-01	  1.2290600e+00 	 2.0823431e-01


.. parsed-literal::

     109	 1.3369392e+00	 1.3733948e-01	 1.3886510e+00	 1.4829237e-01	  1.2300476e+00 	 2.0507765e-01


.. parsed-literal::

     110	 1.3397990e+00	 1.3710269e-01	 1.3916206e+00	 1.4806434e-01	[ 1.2349714e+00]	 2.0901227e-01


.. parsed-literal::

     111	 1.3429793e+00	 1.3672200e-01	 1.3950379e+00	 1.4772549e-01	[ 1.2363293e+00]	 2.1317196e-01


.. parsed-literal::

     112	 1.3457163e+00	 1.3646575e-01	 1.3978513e+00	 1.4735833e-01	[ 1.2397661e+00]	 2.0104933e-01


.. parsed-literal::

     113	 1.3486813e+00	 1.3642324e-01	 1.4008801e+00	 1.4712063e-01	[ 1.2437955e+00]	 2.0916057e-01


.. parsed-literal::

     114	 1.3520697e+00	 1.3593706e-01	 1.4044902e+00	 1.4685620e-01	  1.2433905e+00 	 2.1563792e-01


.. parsed-literal::

     115	 1.3559549e+00	 1.3574504e-01	 1.4084342e+00	 1.4660133e-01	[ 1.2456640e+00]	 2.0382285e-01
     116	 1.3599587e+00	 1.3538641e-01	 1.4125856e+00	 1.4636524e-01	[ 1.2482940e+00]	 1.9833422e-01


.. parsed-literal::

     117	 1.3628836e+00	 1.3512967e-01	 1.4155913e+00	 1.4620793e-01	  1.2473353e+00 	 2.1389318e-01


.. parsed-literal::

     118	 1.3666739e+00	 1.3486804e-01	 1.4193879e+00	 1.4612458e-01	[ 1.2515321e+00]	 2.1319866e-01


.. parsed-literal::

     119	 1.3701886e+00	 1.3460108e-01	 1.4229407e+00	 1.4616970e-01	[ 1.2530515e+00]	 2.2223091e-01


.. parsed-literal::

     120	 1.3725141e+00	 1.3447313e-01	 1.4253377e+00	 1.4624253e-01	  1.2510081e+00 	 2.1930003e-01


.. parsed-literal::

     121	 1.3767265e+00	 1.3416251e-01	 1.4297770e+00	 1.4640504e-01	  1.2375660e+00 	 2.0431042e-01


.. parsed-literal::

     122	 1.3780101e+00	 1.3415802e-01	 1.4313596e+00	 1.4660648e-01	  1.2239240e+00 	 2.0234728e-01


.. parsed-literal::

     123	 1.3817361e+00	 1.3401003e-01	 1.4348410e+00	 1.4633311e-01	  1.2304700e+00 	 2.1623373e-01


.. parsed-literal::

     124	 1.3835401e+00	 1.3381154e-01	 1.4366768e+00	 1.4617861e-01	  1.2291022e+00 	 2.1599483e-01


.. parsed-literal::

     125	 1.3861511e+00	 1.3371581e-01	 1.4393090e+00	 1.4605895e-01	  1.2254601e+00 	 2.0458174e-01
     126	 1.3896923e+00	 1.3363513e-01	 1.4429304e+00	 1.4584837e-01	  1.2208288e+00 	 1.7271185e-01


.. parsed-literal::

     127	 1.3929932e+00	 1.3366980e-01	 1.4463242e+00	 1.4591018e-01	  1.2166779e+00 	 2.0491314e-01
     128	 1.3951289e+00	 1.3369538e-01	 1.4484437e+00	 1.4596958e-01	  1.2181489e+00 	 1.6989040e-01


.. parsed-literal::

     129	 1.3979260e+00	 1.3359817e-01	 1.4513780e+00	 1.4594807e-01	  1.2141528e+00 	 2.0399523e-01


.. parsed-literal::

     130	 1.3999930e+00	 1.3349500e-01	 1.4537215e+00	 1.4585789e-01	  1.2084547e+00 	 2.0395327e-01
     131	 1.4025517e+00	 1.3334279e-01	 1.4562743e+00	 1.4565242e-01	  1.2085817e+00 	 2.0352221e-01


.. parsed-literal::

     132	 1.4047430e+00	 1.3308838e-01	 1.4585695e+00	 1.4529009e-01	  1.2051900e+00 	 2.1151471e-01
     133	 1.4071714e+00	 1.3289858e-01	 1.4610852e+00	 1.4497965e-01	  1.2021114e+00 	 2.0140243e-01


.. parsed-literal::

     134	 1.4102701e+00	 1.3264652e-01	 1.4644950e+00	 1.4445263e-01	  1.1900256e+00 	 2.1851468e-01
     135	 1.4129180e+00	 1.3245394e-01	 1.4671198e+00	 1.4408233e-01	  1.1922700e+00 	 1.8105125e-01


.. parsed-literal::

     136	 1.4143961e+00	 1.3248976e-01	 1.4685174e+00	 1.4411466e-01	  1.1962408e+00 	 2.1726775e-01


.. parsed-literal::

     137	 1.4166215e+00	 1.3236718e-01	 1.4706285e+00	 1.4404842e-01	  1.2015104e+00 	 2.0132184e-01
     138	 1.4189330e+00	 1.3229328e-01	 1.4730246e+00	 1.4361314e-01	  1.2032905e+00 	 1.8219495e-01


.. parsed-literal::

     139	 1.4213934e+00	 1.3193725e-01	 1.4754950e+00	 1.4325138e-01	  1.2050639e+00 	 2.0028377e-01
     140	 1.4239958e+00	 1.3162699e-01	 1.4782470e+00	 1.4280102e-01	  1.2023666e+00 	 2.0011210e-01


.. parsed-literal::

     141	 1.4266531e+00	 1.3142988e-01	 1.4810409e+00	 1.4238905e-01	  1.2024273e+00 	 1.8371630e-01


.. parsed-literal::

     142	 1.4282281e+00	 1.3139524e-01	 1.4828264e+00	 1.4216338e-01	  1.1957639e+00 	 2.0525837e-01


.. parsed-literal::

     143	 1.4305483e+00	 1.3148008e-01	 1.4850162e+00	 1.4221543e-01	  1.2026843e+00 	 2.0581651e-01


.. parsed-literal::

     144	 1.4319435e+00	 1.3162402e-01	 1.4863191e+00	 1.4229609e-01	  1.2064773e+00 	 2.1592689e-01
     145	 1.4334202e+00	 1.3178364e-01	 1.4877249e+00	 1.4239639e-01	  1.2102281e+00 	 1.7144108e-01


.. parsed-literal::

     146	 1.4350889e+00	 1.3194082e-01	 1.4892859e+00	 1.4253836e-01	  1.2102330e+00 	 2.0541382e-01


.. parsed-literal::

     147	 1.4368074e+00	 1.3184558e-01	 1.4910101e+00	 1.4238731e-01	  1.2110972e+00 	 2.0680213e-01
     148	 1.4388916e+00	 1.3158470e-01	 1.4931164e+00	 1.4221032e-01	  1.2084030e+00 	 1.7815924e-01


.. parsed-literal::

     149	 1.4406447e+00	 1.3142620e-01	 1.4948955e+00	 1.4207845e-01	  1.2062768e+00 	 2.1441483e-01


.. parsed-literal::

     150	 1.4427549e+00	 1.3122166e-01	 1.4972399e+00	 1.4196170e-01	  1.1946510e+00 	 2.1206260e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 5s, sys: 1.2 s, total: 2min 6s
    Wall time: 31.7 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f54202e4ca0>



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

    Inserting handle into data store.  model: GPz_model.pkl, gpz_run
    Process 0 running estimator on chunk 0 - 10,000
    Process 0 estimating GPz PZ PDF for rows 0 - 10,000


.. parsed-literal::

    Inserting handle into data store.  output_gpz_run: inprogress_output_gpz_run.hdf5, gpz_run


.. parsed-literal::

    Process 0 running estimator on chunk 10,000 - 20,000
    Process 0 estimating GPz PZ PDF for rows 10,000 - 20,000


.. parsed-literal::

    Process 0 running estimator on chunk 20,000 - 20,449
    Process 0 estimating GPz PZ PDF for rows 20,000 - 20,449
    CPU times: user 1.69 s, sys: 50 ms, total: 1.74 s
    Wall time: 533 ms


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




.. image:: ../../../docs/rendered/estimation_examples/06_GPz_files/../../../docs/rendered/estimation_examples/06_GPz_16_1.png


GPzEstimator parameterizes each PDF as a single Gaussian, here we see a
few examples of Gaussians of different widths. Now let’s grab the mode
of each PDF (stored as ancil data in the ensemble) and compare to the
true redshifts from the test_data file:

.. code:: ipython3

    truez = test_data.data['photometry']['redshift']
    zmode = ens.ancil['zmode'].flatten()

.. code:: ipython3

    plt.figure(figsize=(12,12))
    plt.scatter(truez, zmode, s=3)
    plt.plot([0,3],[0,3], 'k--')
    plt.xlabel("redshift", fontsize=12)
    plt.ylabel("z mode", fontsize=12)




.. parsed-literal::

    Text(0, 0.5, 'z mode')




.. image:: ../../../docs/rendered/estimation_examples/06_GPz_files/../../../docs/rendered/estimation_examples/06_GPz_19_1.png

