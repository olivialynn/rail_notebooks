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
       1	-3.3851491e-01	 3.1936032e-01	-3.2876774e-01	 3.2574520e-01	[-3.4048243e-01]	 4.7033215e-01


.. parsed-literal::

       2	-2.6899678e-01	 3.0883997e-01	-2.4541198e-01	 3.1529802e-01	[-2.6240540e-01]	 2.3508263e-01


.. parsed-literal::

       3	-2.2321524e-01	 2.8746155e-01	-1.8069943e-01	 2.9043387e-01	[-1.9186288e-01]	 2.8486753e-01
       4	-1.9655355e-01	 2.6386564e-01	-1.5725994e-01	 2.6736942e-01	[-1.8015425e-01]	 1.8896532e-01


.. parsed-literal::

       5	-9.8191666e-02	 2.5566409e-01	-6.3396782e-02	 2.6280703e-01	[-8.9285472e-02]	 1.8967915e-01


.. parsed-literal::

       6	-6.6566041e-02	 2.5051135e-01	-3.5226872e-02	 2.5507243e-01	[-5.2283253e-02]	 2.1092248e-01


.. parsed-literal::

       7	-4.7543172e-02	 2.4773109e-01	-2.4016394e-02	 2.5258682e-01	[-4.1385663e-02]	 2.1330786e-01


.. parsed-literal::

       8	-3.6252174e-02	 2.4584773e-01	-1.6163480e-02	 2.5077133e-01	[-3.3335128e-02]	 2.2138977e-01


.. parsed-literal::

       9	-2.2188300e-02	 2.4325025e-01	-4.9975390e-03	 2.4828603e-01	[-2.3691520e-02]	 2.1975684e-01


.. parsed-literal::

      10	-1.4902635e-02	 2.4207583e-01	 3.8801068e-04	 2.4945514e-01	 -2.7815369e-02 	 2.1544242e-01


.. parsed-literal::

      11	-8.8702939e-03	 2.4084260e-01	 5.7539800e-03	 2.4751265e-01	[-2.1834067e-02]	 2.2241998e-01


.. parsed-literal::

      12	-6.6026865e-03	 2.4039598e-01	 7.9084432e-03	 2.4666786e-01	[-1.7051109e-02]	 2.0743537e-01


.. parsed-literal::

      13	-2.6437388e-03	 2.3961112e-01	 1.1807003e-02	 2.4536328e-01	[-1.0834790e-02]	 2.2106934e-01


.. parsed-literal::

      14	 3.7788383e-03	 2.3826683e-01	 1.8760890e-02	 2.4326171e-01	[-1.0146614e-03]	 2.0835328e-01


.. parsed-literal::

      15	 6.2311417e-02	 2.2077983e-01	 8.1229659e-02	 2.3384356e-01	[ 5.9643951e-02]	 3.0960655e-01
      16	 1.4479598e-01	 2.1726237e-01	 1.6459741e-01	 2.2792052e-01	[ 1.3961276e-01]	 2.0401788e-01


.. parsed-literal::

      17	 1.7843124e-01	 2.1289261e-01	 2.0649656e-01	 2.2205343e-01	[ 1.4442718e-01]	 1.9210052e-01


.. parsed-literal::

      18	 2.6046672e-01	 2.1313681e-01	 2.8898897e-01	 2.2053607e-01	[ 2.4372076e-01]	 2.2064400e-01


.. parsed-literal::

      19	 2.9119955e-01	 2.1015805e-01	 3.1909527e-01	 2.1804297e-01	[ 2.7526021e-01]	 2.0684457e-01


.. parsed-literal::

      20	 3.3248182e-01	 2.0658759e-01	 3.6144310e-01	 2.1517574e-01	[ 3.1347495e-01]	 2.2135162e-01


.. parsed-literal::

      21	 4.3181845e-01	 2.0141567e-01	 4.6249468e-01	 2.1027718e-01	[ 4.0507281e-01]	 2.1686363e-01


.. parsed-literal::

      22	 5.3111832e-01	 2.0149458e-01	 5.6865134e-01	 2.0917692e-01	[ 5.1527389e-01]	 2.1331716e-01


.. parsed-literal::

      23	 5.7668657e-01	 2.0088304e-01	 6.1603311e-01	 2.1158573e-01	[ 5.4975237e-01]	 2.1124268e-01
      24	 6.1974594e-01	 1.9899131e-01	 6.5721350e-01	 2.0840819e-01	[ 5.8821083e-01]	 1.7946553e-01


.. parsed-literal::

      25	 6.5364530e-01	 1.9498667e-01	 6.8963219e-01	 2.0668896e-01	[ 6.1647310e-01]	 2.0542049e-01


.. parsed-literal::

      26	 6.7441325e-01	 1.9466922e-01	 7.0974643e-01	 2.0689376e-01	[ 6.3062844e-01]	 2.1318126e-01


.. parsed-literal::

      27	 7.0645391e-01	 1.9533876e-01	 7.4114722e-01	 2.0636557e-01	[ 6.6728934e-01]	 2.2022271e-01
      28	 7.3090594e-01	 1.9917495e-01	 7.6607655e-01	 2.0774201e-01	[ 6.8898664e-01]	 1.9012165e-01


.. parsed-literal::

      29	 7.5278779e-01	 1.9973735e-01	 7.8951484e-01	 2.0917449e-01	[ 7.0781649e-01]	 2.2147822e-01


.. parsed-literal::

      30	 7.6815147e-01	 2.0088065e-01	 8.0509862e-01	 2.1024485e-01	[ 7.3156787e-01]	 2.2027850e-01


.. parsed-literal::

      31	 7.8665439e-01	 2.0149757e-01	 8.2403617e-01	 2.1127124e-01	[ 7.5255856e-01]	 2.0589018e-01


.. parsed-literal::

      32	 8.0986297e-01	 2.0269999e-01	 8.4712796e-01	 2.1421233e-01	[ 7.7378017e-01]	 2.1828723e-01


.. parsed-literal::

      33	 8.3075640e-01	 2.0411910e-01	 8.7096950e-01	 2.1077692e-01	[ 7.9300431e-01]	 2.1566749e-01


.. parsed-literal::

      34	 8.5940337e-01	 2.0150324e-01	 8.9962527e-01	 2.1076591e-01	[ 8.2005000e-01]	 2.1918297e-01


.. parsed-literal::

      35	 8.7583005e-01	 1.9851406e-01	 9.1580048e-01	 2.0912797e-01	[ 8.3463502e-01]	 2.1124411e-01


.. parsed-literal::

      36	 8.8858785e-01	 1.9686310e-01	 9.2866182e-01	 2.0780527e-01	[ 8.4706296e-01]	 2.1338916e-01
      37	 9.0603931e-01	 1.9463454e-01	 9.4701009e-01	 2.0420713e-01	[ 8.6881819e-01]	 1.9166470e-01


.. parsed-literal::

      38	 9.1629264e-01	 1.9495039e-01	 9.5797739e-01	 2.0573980e-01	[ 8.7515069e-01]	 2.0664096e-01


.. parsed-literal::

      39	 9.3059569e-01	 1.9269363e-01	 9.7239709e-01	 2.0294506e-01	[ 8.9015833e-01]	 2.1381927e-01


.. parsed-literal::

      40	 9.4484320e-01	 1.9002556e-01	 9.8696531e-01	 2.0029237e-01	[ 9.0223700e-01]	 2.1439958e-01
      41	 9.5714720e-01	 1.8870472e-01	 9.9949716e-01	 2.0028158e-01	[ 9.0970913e-01]	 1.7972684e-01


.. parsed-literal::

      42	 9.6751071e-01	 1.8788956e-01	 1.0111580e+00	 2.0131526e-01	[ 9.1113910e-01]	 1.9997406e-01
      43	 9.7705861e-01	 1.8635877e-01	 1.0210952e+00	 1.9832456e-01	[ 9.1630030e-01]	 1.9721866e-01


.. parsed-literal::

      44	 9.8594389e-01	 1.8594474e-01	 1.0292664e+00	 1.9772870e-01	[ 9.3033816e-01]	 2.0840526e-01
      45	 9.9223369e-01	 1.8545517e-01	 1.0356824e+00	 1.9718327e-01	[ 9.3692085e-01]	 1.8979597e-01


.. parsed-literal::

      46	 1.0065998e+00	 1.8375502e-01	 1.0506893e+00	 1.9461550e-01	[ 9.5167520e-01]	 2.1642685e-01


.. parsed-literal::

      47	 1.0142879e+00	 1.8197319e-01	 1.0591274e+00	 1.9161897e-01	[ 9.5845927e-01]	 2.2049785e-01


.. parsed-literal::

      48	 1.0237297e+00	 1.8119050e-01	 1.0684678e+00	 1.9003550e-01	[ 9.6934138e-01]	 2.0645571e-01


.. parsed-literal::

      49	 1.0294184e+00	 1.8082073e-01	 1.0738683e+00	 1.8929660e-01	[ 9.7525693e-01]	 2.1272302e-01


.. parsed-literal::

      50	 1.0373434e+00	 1.8000045e-01	 1.0817673e+00	 1.8768891e-01	[ 9.8058632e-01]	 2.1093202e-01
      51	 1.0488644e+00	 1.7820006e-01	 1.0935048e+00	 1.8478987e-01	[ 9.8603515e-01]	 1.9000769e-01


.. parsed-literal::

      52	 1.0539674e+00	 1.7714717e-01	 1.0986809e+00	 1.8398956e-01	[ 9.9020782e-01]	 3.3273196e-01


.. parsed-literal::

      53	 1.0597957e+00	 1.7612232e-01	 1.1046620e+00	 1.8273106e-01	[ 9.9412876e-01]	 2.0768809e-01


.. parsed-literal::

      54	 1.0649748e+00	 1.7497811e-01	 1.1098219e+00	 1.8190499e-01	[ 1.0030618e+00]	 2.0375490e-01


.. parsed-literal::

      55	 1.0688457e+00	 1.7380992e-01	 1.1138734e+00	 1.8058300e-01	[ 1.0111037e+00]	 2.0353961e-01
      56	 1.0724243e+00	 1.7352048e-01	 1.1172765e+00	 1.8053008e-01	[ 1.0154850e+00]	 1.9689941e-01


.. parsed-literal::

      57	 1.0777671e+00	 1.7269719e-01	 1.1226048e+00	 1.7960717e-01	[ 1.0212978e+00]	 1.8998671e-01
      58	 1.0837310e+00	 1.7165822e-01	 1.1287235e+00	 1.7833943e-01	[ 1.0283691e+00]	 1.7881632e-01


.. parsed-literal::

      59	 1.0890774e+00	 1.6917857e-01	 1.1345542e+00	 1.7536407e-01	[ 1.0348823e+00]	 2.0894599e-01


.. parsed-literal::

      60	 1.0967316e+00	 1.6838322e-01	 1.1422063e+00	 1.7495264e-01	[ 1.0442549e+00]	 2.0908332e-01
      61	 1.1006780e+00	 1.6774094e-01	 1.1461291e+00	 1.7484166e-01	[ 1.0483071e+00]	 1.7831683e-01


.. parsed-literal::

      62	 1.1085402e+00	 1.6601197e-01	 1.1543456e+00	 1.7409240e-01	[ 1.0517843e+00]	 2.1988821e-01


.. parsed-literal::

      63	 1.1102012e+00	 1.6557084e-01	 1.1567079e+00	 1.7468523e-01	  1.0487761e+00 	 2.0929527e-01


.. parsed-literal::

      64	 1.1181113e+00	 1.6423293e-01	 1.1644296e+00	 1.7356222e-01	[ 1.0552469e+00]	 2.0414186e-01
      65	 1.1208356e+00	 1.6401525e-01	 1.1670927e+00	 1.7332582e-01	[ 1.0583158e+00]	 1.7396474e-01


.. parsed-literal::

      66	 1.1258726e+00	 1.6347702e-01	 1.1722389e+00	 1.7258995e-01	[ 1.0634971e+00]	 2.1368337e-01
      67	 1.1327629e+00	 1.6284787e-01	 1.1792843e+00	 1.7159680e-01	[ 1.0731317e+00]	 2.0071435e-01


.. parsed-literal::

      68	 1.1353067e+00	 1.6362993e-01	 1.1823987e+00	 1.7132016e-01	  1.0717107e+00 	 2.0717263e-01


.. parsed-literal::

      69	 1.1431445e+00	 1.6302133e-01	 1.1900032e+00	 1.7103802e-01	[ 1.0824314e+00]	 2.0339942e-01


.. parsed-literal::

      70	 1.1467101e+00	 1.6272806e-01	 1.1936560e+00	 1.7097388e-01	[ 1.0847226e+00]	 2.0713019e-01


.. parsed-literal::

      71	 1.1523097e+00	 1.6249680e-01	 1.1996043e+00	 1.7122036e-01	[ 1.0870005e+00]	 2.0307231e-01
      72	 1.1571909e+00	 1.6272896e-01	 1.2048368e+00	 1.7166480e-01	[ 1.0894325e+00]	 2.1065593e-01


.. parsed-literal::

      73	 1.1625734e+00	 1.6208125e-01	 1.2101572e+00	 1.7168073e-01	[ 1.0931768e+00]	 1.8608212e-01


.. parsed-literal::

      74	 1.1670188e+00	 1.6114429e-01	 1.2145794e+00	 1.7134778e-01	[ 1.0981929e+00]	 2.1428943e-01


.. parsed-literal::

      75	 1.1706967e+00	 1.6021388e-01	 1.2183112e+00	 1.7090027e-01	[ 1.1027834e+00]	 2.1352744e-01


.. parsed-literal::

      76	 1.1763557e+00	 1.5782873e-01	 1.2245056e+00	 1.6931118e-01	[ 1.1105532e+00]	 2.1227503e-01


.. parsed-literal::

      77	 1.1823580e+00	 1.5657152e-01	 1.2304854e+00	 1.6867373e-01	[ 1.1153931e+00]	 2.1560431e-01


.. parsed-literal::

      78	 1.1865457e+00	 1.5604921e-01	 1.2347094e+00	 1.6823585e-01	[ 1.1173416e+00]	 2.1903348e-01


.. parsed-literal::

      79	 1.1920224e+00	 1.5483360e-01	 1.2403208e+00	 1.6700117e-01	[ 1.1205961e+00]	 2.1101236e-01
      80	 1.1980947e+00	 1.5361726e-01	 1.2467118e+00	 1.6555458e-01	[ 1.1236520e+00]	 1.9290376e-01


.. parsed-literal::

      81	 1.2019650e+00	 1.5203119e-01	 1.2507778e+00	 1.6374749e-01	[ 1.1291528e+00]	 2.0511889e-01
      82	 1.2066966e+00	 1.5195875e-01	 1.2553034e+00	 1.6377449e-01	[ 1.1341529e+00]	 1.8842340e-01


.. parsed-literal::

      83	 1.2107281e+00	 1.5152025e-01	 1.2593695e+00	 1.6358993e-01	[ 1.1389118e+00]	 2.1710157e-01
      84	 1.2157141e+00	 1.5090467e-01	 1.2645701e+00	 1.6342210e-01	[ 1.1444824e+00]	 1.8832994e-01


.. parsed-literal::

      85	 1.2202879e+00	 1.4985970e-01	 1.2696436e+00	 1.6235350e-01	[ 1.1479298e+00]	 1.8155551e-01
      86	 1.2246022e+00	 1.4956987e-01	 1.2740061e+00	 1.6229501e-01	[ 1.1513674e+00]	 1.8412852e-01


.. parsed-literal::

      87	 1.2277640e+00	 1.4937991e-01	 1.2771333e+00	 1.6197456e-01	[ 1.1544055e+00]	 2.0715547e-01


.. parsed-literal::

      88	 1.2331288e+00	 1.4861128e-01	 1.2826880e+00	 1.6074894e-01	[ 1.1603069e+00]	 2.1638632e-01


.. parsed-literal::

      89	 1.2359364e+00	 1.4832145e-01	 1.2858839e+00	 1.6018443e-01	[ 1.1640014e+00]	 2.1081281e-01
      90	 1.2407987e+00	 1.4812027e-01	 1.2904863e+00	 1.5992053e-01	[ 1.1701995e+00]	 1.7657185e-01


.. parsed-literal::

      91	 1.2430499e+00	 1.4790525e-01	 1.2927389e+00	 1.5972574e-01	[ 1.1742333e+00]	 1.9817352e-01
      92	 1.2471580e+00	 1.4750621e-01	 1.2968820e+00	 1.5919636e-01	[ 1.1790755e+00]	 1.9328785e-01


.. parsed-literal::

      93	 1.2512070e+00	 1.4649371e-01	 1.3011881e+00	 1.5783237e-01	[ 1.1907002e+00]	 1.9186282e-01


.. parsed-literal::

      94	 1.2566025e+00	 1.4564640e-01	 1.3066160e+00	 1.5724705e-01	[ 1.1908627e+00]	 2.2251916e-01


.. parsed-literal::

      95	 1.2596331e+00	 1.4503985e-01	 1.3097058e+00	 1.5663543e-01	  1.1903545e+00 	 2.1301436e-01


.. parsed-literal::

      96	 1.2639141e+00	 1.4403499e-01	 1.3140692e+00	 1.5558495e-01	[ 1.1922185e+00]	 2.0864940e-01
      97	 1.2693488e+00	 1.4276280e-01	 1.3195929e+00	 1.5473530e-01	[ 1.1943270e+00]	 2.0252919e-01


.. parsed-literal::

      98	 1.2756622e+00	 1.4168200e-01	 1.3259532e+00	 1.5313578e-01	[ 1.2047690e+00]	 2.2012043e-01


.. parsed-literal::

      99	 1.2799263e+00	 1.4150982e-01	 1.3301359e+00	 1.5286978e-01	[ 1.2106088e+00]	 2.1339774e-01
     100	 1.2851455e+00	 1.4115908e-01	 1.3354495e+00	 1.5203877e-01	[ 1.2185680e+00]	 2.0139122e-01


.. parsed-literal::

     101	 1.2892935e+00	 1.4093991e-01	 1.3398196e+00	 1.5149707e-01	  1.2184137e+00 	 2.1597934e-01


.. parsed-literal::

     102	 1.2933201e+00	 1.4066813e-01	 1.3439610e+00	 1.5082745e-01	[ 1.2207625e+00]	 2.1225309e-01


.. parsed-literal::

     103	 1.3001703e+00	 1.4045056e-01	 1.3510180e+00	 1.5011386e-01	  1.2193062e+00 	 2.1002769e-01


.. parsed-literal::

     104	 1.3041704e+00	 1.3997534e-01	 1.3551957e+00	 1.4945743e-01	[ 1.2211299e+00]	 2.0454669e-01


.. parsed-literal::

     105	 1.3081519e+00	 1.3941000e-01	 1.3591977e+00	 1.4916541e-01	[ 1.2212644e+00]	 2.0936179e-01


.. parsed-literal::

     106	 1.3123644e+00	 1.3866737e-01	 1.3634555e+00	 1.4881712e-01	  1.2209383e+00 	 2.0479107e-01


.. parsed-literal::

     107	 1.3164342e+00	 1.3800925e-01	 1.3676890e+00	 1.4844073e-01	  1.2204039e+00 	 2.1793675e-01


.. parsed-literal::

     108	 1.3204224e+00	 1.3764204e-01	 1.3717347e+00	 1.4827801e-01	[ 1.2237325e+00]	 2.1352243e-01


.. parsed-literal::

     109	 1.3244129e+00	 1.3732603e-01	 1.3757700e+00	 1.4830197e-01	[ 1.2255028e+00]	 2.1337342e-01
     110	 1.3274802e+00	 1.3713259e-01	 1.3790460e+00	 1.4847127e-01	[ 1.2277790e+00]	 1.9771004e-01


.. parsed-literal::

     111	 1.3309899e+00	 1.3684759e-01	 1.3825012e+00	 1.4821100e-01	[ 1.2329431e+00]	 2.0949960e-01


.. parsed-literal::

     112	 1.3338100e+00	 1.3657614e-01	 1.3853036e+00	 1.4780396e-01	[ 1.2361714e+00]	 2.1536851e-01


.. parsed-literal::

     113	 1.3382000e+00	 1.3609436e-01	 1.3898307e+00	 1.4727763e-01	[ 1.2398641e+00]	 2.0432448e-01
     114	 1.3422522e+00	 1.3522744e-01	 1.3942106e+00	 1.4534791e-01	[ 1.2434835e+00]	 2.0257735e-01


.. parsed-literal::

     115	 1.3475184e+00	 1.3471052e-01	 1.3996568e+00	 1.4576869e-01	[ 1.2437504e+00]	 2.1897149e-01
     116	 1.3503953e+00	 1.3447858e-01	 1.4025070e+00	 1.4568264e-01	[ 1.2457103e+00]	 1.8936372e-01


.. parsed-literal::

     117	 1.3533553e+00	 1.3415994e-01	 1.4055585e+00	 1.4549960e-01	[ 1.2466875e+00]	 2.0759320e-01


.. parsed-literal::

     118	 1.3572467e+00	 1.3364834e-01	 1.4097701e+00	 1.4524404e-01	[ 1.2472599e+00]	 2.2076344e-01


.. parsed-literal::

     119	 1.3608208e+00	 1.3338814e-01	 1.4134354e+00	 1.4497058e-01	[ 1.2515115e+00]	 2.0549273e-01


.. parsed-literal::

     120	 1.3629224e+00	 1.3323299e-01	 1.4155475e+00	 1.4476523e-01	[ 1.2543779e+00]	 2.0384932e-01


.. parsed-literal::

     121	 1.3660270e+00	 1.3301336e-01	 1.4186427e+00	 1.4437801e-01	[ 1.2608542e+00]	 2.1291113e-01
     122	 1.3691752e+00	 1.3278819e-01	 1.4219068e+00	 1.4407397e-01	[ 1.2635659e+00]	 1.9013619e-01


.. parsed-literal::

     123	 1.3728224e+00	 1.3242367e-01	 1.4255744e+00	 1.4366068e-01	[ 1.2695261e+00]	 2.2093630e-01
     124	 1.3761247e+00	 1.3245971e-01	 1.4288932e+00	 1.4344284e-01	[ 1.2711320e+00]	 1.9496369e-01


.. parsed-literal::

     125	 1.3794934e+00	 1.3207157e-01	 1.4323769e+00	 1.4298877e-01	  1.2700234e+00 	 2.0073557e-01


.. parsed-literal::

     126	 1.3822203e+00	 1.3205373e-01	 1.4351292e+00	 1.4289097e-01	  1.2704889e+00 	 2.1115994e-01


.. parsed-literal::

     127	 1.3848048e+00	 1.3189150e-01	 1.4376781e+00	 1.4232668e-01	[ 1.2713218e+00]	 2.0902467e-01


.. parsed-literal::

     128	 1.3872351e+00	 1.3175141e-01	 1.4401176e+00	 1.4214445e-01	[ 1.2719608e+00]	 2.0848250e-01
     129	 1.3903802e+00	 1.3148916e-01	 1.4432879e+00	 1.4191178e-01	[ 1.2746427e+00]	 1.8303704e-01


.. parsed-literal::

     130	 1.3920340e+00	 1.3082611e-01	 1.4451195e+00	 1.4159975e-01	  1.2681313e+00 	 2.4431944e-01
     131	 1.3955525e+00	 1.3087830e-01	 1.4485871e+00	 1.4185648e-01	  1.2720478e+00 	 1.8640447e-01


.. parsed-literal::

     132	 1.3973386e+00	 1.3087970e-01	 1.4503681e+00	 1.4199297e-01	  1.2725071e+00 	 2.0806313e-01


.. parsed-literal::

     133	 1.3998733e+00	 1.3078214e-01	 1.4530119e+00	 1.4212689e-01	  1.2694926e+00 	 2.1391201e-01


.. parsed-literal::

     134	 1.4043788e+00	 1.3054222e-01	 1.4577075e+00	 1.4224799e-01	  1.2651554e+00 	 2.0985293e-01
     135	 1.4062359e+00	 1.3064585e-01	 1.4601676e+00	 1.4256052e-01	  1.2529045e+00 	 1.9033265e-01


.. parsed-literal::

     136	 1.4117789e+00	 1.3024007e-01	 1.4654436e+00	 1.4206871e-01	  1.2607260e+00 	 2.1307778e-01
     137	 1.4134677e+00	 1.3002955e-01	 1.4670444e+00	 1.4163693e-01	  1.2651775e+00 	 1.8697786e-01


.. parsed-literal::

     138	 1.4158198e+00	 1.2992975e-01	 1.4694735e+00	 1.4113928e-01	  1.2673530e+00 	 2.0977330e-01


.. parsed-literal::

     139	 1.4184074e+00	 1.2967355e-01	 1.4721571e+00	 1.4066443e-01	  1.2694689e+00 	 2.0204639e-01
     140	 1.4206186e+00	 1.2966435e-01	 1.4744636e+00	 1.4058132e-01	  1.2689839e+00 	 2.0190239e-01


.. parsed-literal::

     141	 1.4239882e+00	 1.2971839e-01	 1.4780717e+00	 1.4054772e-01	  1.2662163e+00 	 2.1016145e-01


.. parsed-literal::

     142	 1.4245666e+00	 1.2962402e-01	 1.4788399e+00	 1.4050657e-01	  1.2631611e+00 	 2.2065258e-01


.. parsed-literal::

     143	 1.4273184e+00	 1.2956403e-01	 1.4813951e+00	 1.4044687e-01	  1.2687464e+00 	 2.0386982e-01


.. parsed-literal::

     144	 1.4286895e+00	 1.2948958e-01	 1.4827356e+00	 1.4035760e-01	  1.2712926e+00 	 2.1949148e-01
     145	 1.4307784e+00	 1.2933028e-01	 1.4848047e+00	 1.4018383e-01	[ 1.2757567e+00]	 1.8025732e-01


.. parsed-literal::

     146	 1.4340364e+00	 1.2911491e-01	 1.4881037e+00	 1.3982285e-01	[ 1.2804475e+00]	 2.0968390e-01


.. parsed-literal::

     147	 1.4348150e+00	 1.2914953e-01	 1.4891609e+00	 1.3972748e-01	[ 1.2816510e+00]	 2.1438217e-01
     148	 1.4399634e+00	 1.2890524e-01	 1.4941983e+00	 1.3938933e-01	[ 1.2866629e+00]	 1.8260336e-01


.. parsed-literal::

     149	 1.4412496e+00	 1.2881285e-01	 1.4954643e+00	 1.3923948e-01	  1.2864531e+00 	 1.7948985e-01


.. parsed-literal::

     150	 1.4434492e+00	 1.2876066e-01	 1.4977362e+00	 1.3910386e-01	  1.2856644e+00 	 2.1349621e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 5s, sys: 1.23 s, total: 2min 6s
    Wall time: 31.7 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f1fddde0100>



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
    CPU times: user 1.7 s, sys: 35 ms, total: 1.74 s
    Wall time: 531 ms


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

