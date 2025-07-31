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
       1	-3.3991726e-01	 3.1934360e-01	-3.3029842e-01	 3.2581052e-01	[-3.4292976e-01]	 4.7816372e-01


.. parsed-literal::

       2	-2.6946333e-01	 3.0912114e-01	-2.4602946e-01	 3.1358088e-01	[-2.6008792e-01]	 2.4157143e-01


.. parsed-literal::

       3	-2.2022218e-01	 2.8569241e-01	-1.7535423e-01	 2.8701560e-01	[-1.8369256e-01]	 3.0071068e-01


.. parsed-literal::

       4	-1.8070425e-01	 2.6822527e-01	-1.3148931e-01	 2.7116560e-01	[-1.4763469e-01]	 3.3459520e-01


.. parsed-literal::

       5	-1.1337741e-01	 2.5609633e-01	-8.2882778e-02	 2.5912731e-01	[-1.0053373e-01]	 2.2260952e-01


.. parsed-literal::

       6	-6.7653389e-02	 2.5172204e-01	-4.0465586e-02	 2.5694524e-01	[-6.0128882e-02]	 2.2060752e-01


.. parsed-literal::

       7	-4.8437993e-02	 2.4792911e-01	-2.5026718e-02	 2.5316173e-01	[-4.5819811e-02]	 2.2705960e-01


.. parsed-literal::

       8	-3.5827138e-02	 2.4579511e-01	-1.5912439e-02	 2.5169611e-01	[-4.1580796e-02]	 2.2141647e-01


.. parsed-literal::

       9	-2.4104933e-02	 2.4348858e-01	-5.9388643e-03	 2.4914825e-01	[-2.8743257e-02]	 2.2528291e-01
      10	-1.3086376e-02	 2.4154052e-01	 2.7966791e-03	 2.4796162e-01	[-2.4486204e-02]	 1.8753624e-01


.. parsed-literal::

      11	-8.0999032e-03	 2.4055609e-01	 7.1323076e-03	 2.4679683e-01	[-1.9564389e-02]	 2.2045350e-01


.. parsed-literal::

      12	-1.9612322e-03	 2.3952592e-01	 1.2286972e-02	 2.4571717e-01	[-1.4184496e-02]	 2.1538329e-01


.. parsed-literal::

      13	 2.6172383e-03	 2.3899687e-01	 1.6148735e-02	 2.4535339e-01	[-1.0962219e-02]	 2.1378565e-01


.. parsed-literal::

      14	 7.7236113e-02	 2.2579493e-01	 9.7658553e-02	 2.3281622e-01	[ 7.8293288e-02]	 5.1254559e-01


.. parsed-literal::

      15	 1.2655333e-01	 2.2244943e-01	 1.4816052e-01	 2.3000035e-01	[ 1.2483884e-01]	 2.2756004e-01


.. parsed-literal::

      16	 1.8008741e-01	 2.1778373e-01	 2.0772631e-01	 2.2343217e-01	[ 1.8346525e-01]	 2.1301627e-01
      17	 2.2890333e-01	 2.1865340e-01	 2.5488771e-01	 2.2434739e-01	[ 2.3476541e-01]	 1.8374515e-01


.. parsed-literal::

      18	 2.6082642e-01	 2.2078805e-01	 2.8980909e-01	 2.2684415e-01	[ 2.7571431e-01]	 2.1646976e-01


.. parsed-literal::

      19	 3.0430955e-01	 2.1545390e-01	 3.3602546e-01	 2.2216002e-01	[ 3.1926194e-01]	 2.1210694e-01


.. parsed-literal::

      20	 3.6174012e-01	 2.1132559e-01	 3.9502481e-01	 2.2138949e-01	[ 3.7295359e-01]	 2.1490788e-01


.. parsed-literal::

      21	 4.2318189e-01	 2.0710385e-01	 4.5832248e-01	 2.1297889e-01	[ 4.3815474e-01]	 2.0849514e-01


.. parsed-literal::

      22	 4.6684601e-01	 2.0318155e-01	 5.0306829e-01	 2.0912144e-01	[ 4.8145156e-01]	 2.1510029e-01


.. parsed-literal::

      23	 5.0569777e-01	 2.0199857e-01	 5.4246012e-01	 2.0689179e-01	[ 5.2278008e-01]	 2.1607041e-01


.. parsed-literal::

      24	 5.8845926e-01	 1.9887808e-01	 6.2640019e-01	 2.0265000e-01	[ 6.0777187e-01]	 2.1387029e-01
      25	 6.3808985e-01	 1.9828215e-01	 6.7785718e-01	 2.0315748e-01	[ 6.4110484e-01]	 1.8705559e-01


.. parsed-literal::

      26	 6.7810935e-01	 1.8963180e-01	 7.1732216e-01	 1.9606073e-01	[ 6.7814520e-01]	 2.2142458e-01


.. parsed-literal::

      27	 6.9985704e-01	 1.9193707e-01	 7.4160100e-01	 1.9899025e-01	[ 6.9034526e-01]	 2.2300816e-01


.. parsed-literal::

      28	 7.4870609e-01	 1.9587897e-01	 7.8859115e-01	 2.0378462e-01	[ 7.5566595e-01]	 2.2154689e-01


.. parsed-literal::

      29	 7.9302937e-01	 1.9818785e-01	 8.3153378e-01	 2.0722025e-01	[ 8.0594130e-01]	 2.2218800e-01


.. parsed-literal::

      30	 8.2741512e-01	 1.9424323e-01	 8.6669339e-01	 2.0570123e-01	[ 8.4364022e-01]	 2.1507645e-01


.. parsed-literal::

      31	 8.5575995e-01	 1.9221434e-01	 8.9607903e-01	 2.0364932e-01	[ 8.7246106e-01]	 2.1205020e-01


.. parsed-literal::

      32	 8.7484987e-01	 1.9155450e-01	 9.1650717e-01	 2.0295335e-01	[ 8.8669121e-01]	 2.2327566e-01


.. parsed-literal::

      33	 8.9120558e-01	 1.8862006e-01	 9.3343244e-01	 1.9986832e-01	[ 8.9909646e-01]	 2.1249819e-01


.. parsed-literal::

      34	 9.0925742e-01	 1.8414523e-01	 9.5189638e-01	 1.9535234e-01	[ 9.0952802e-01]	 2.1494174e-01


.. parsed-literal::

      35	 9.3165796e-01	 1.8056284e-01	 9.7428702e-01	 1.9051151e-01	[ 9.2668360e-01]	 2.2270727e-01


.. parsed-literal::

      36	 9.5109255e-01	 1.7848780e-01	 9.9384693e-01	 1.8645958e-01	[ 9.4788630e-01]	 2.0649958e-01


.. parsed-literal::

      37	 9.6747675e-01	 1.7576501e-01	 1.0109261e+00	 1.8348340e-01	[ 9.6166396e-01]	 2.0707297e-01


.. parsed-literal::

      38	 9.8626468e-01	 1.7227427e-01	 1.0310898e+00	 1.7936742e-01	[ 9.7389569e-01]	 2.2241902e-01
      39	 1.0064839e+00	 1.6877729e-01	 1.0524013e+00	 1.7583406e-01	[ 9.8653477e-01]	 1.8413448e-01


.. parsed-literal::

      40	 1.0252939e+00	 1.6561557e-01	 1.0720643e+00	 1.7253372e-01	[ 9.9538976e-01]	 2.1303940e-01


.. parsed-literal::

      41	 1.0412159e+00	 1.6345725e-01	 1.0884687e+00	 1.7002446e-01	[ 1.0117294e+00]	 2.1298170e-01


.. parsed-literal::

      42	 1.0580480e+00	 1.6219190e-01	 1.1049871e+00	 1.6918393e-01	[ 1.0358645e+00]	 2.1773195e-01


.. parsed-literal::

      43	 1.0749531e+00	 1.5972072e-01	 1.1226770e+00	 1.6858627e-01	[ 1.0475960e+00]	 2.1479511e-01


.. parsed-literal::

      44	 1.0872424e+00	 1.5776909e-01	 1.1345017e+00	 1.6666953e-01	[ 1.0600932e+00]	 2.1069670e-01
      45	 1.0945185e+00	 1.5627387e-01	 1.1416393e+00	 1.6546446e-01	[ 1.0641620e+00]	 2.0985365e-01


.. parsed-literal::

      46	 1.1126347e+00	 1.5162757e-01	 1.1598700e+00	 1.6184173e-01	[ 1.0712575e+00]	 2.1323442e-01


.. parsed-literal::

      47	 1.1212520e+00	 1.4769588e-01	 1.1691637e+00	 1.5918902e-01	  1.0621852e+00 	 2.1477032e-01
      48	 1.1326750e+00	 1.4681848e-01	 1.1804006e+00	 1.5805539e-01	[ 1.0786770e+00]	 1.9797945e-01


.. parsed-literal::

      49	 1.1433014e+00	 1.4543590e-01	 1.1912071e+00	 1.5687231e-01	[ 1.0885152e+00]	 2.0208025e-01


.. parsed-literal::

      50	 1.1522099e+00	 1.4437104e-01	 1.2003583e+00	 1.5606693e-01	[ 1.0948887e+00]	 2.1356130e-01


.. parsed-literal::

      51	 1.1715224e+00	 1.4279198e-01	 1.2199942e+00	 1.5570871e-01	[ 1.1072647e+00]	 2.1817088e-01
      52	 1.1758187e+00	 1.4181336e-01	 1.2246706e+00	 1.5426919e-01	[ 1.1111320e+00]	 2.0192575e-01


.. parsed-literal::

      53	 1.1915170e+00	 1.4140995e-01	 1.2400267e+00	 1.5336649e-01	[ 1.1319418e+00]	 2.1463728e-01


.. parsed-literal::

      54	 1.2005349e+00	 1.4015693e-01	 1.2490206e+00	 1.5238423e-01	[ 1.1377063e+00]	 2.1229386e-01
      55	 1.2094313e+00	 1.3921569e-01	 1.2580574e+00	 1.5095113e-01	[ 1.1442806e+00]	 1.8672776e-01


.. parsed-literal::

      56	 1.2191261e+00	 1.3810171e-01	 1.2680933e+00	 1.4965076e-01	[ 1.1517919e+00]	 2.1819806e-01


.. parsed-literal::

      57	 1.2284146e+00	 1.3723644e-01	 1.2773853e+00	 1.4855741e-01	[ 1.1595571e+00]	 2.0478559e-01


.. parsed-literal::

      58	 1.2363414e+00	 1.3647001e-01	 1.2855200e+00	 1.4773279e-01	[ 1.1655815e+00]	 2.0348620e-01


.. parsed-literal::

      59	 1.2475794e+00	 1.3510336e-01	 1.2973503e+00	 1.4649785e-01	[ 1.1733803e+00]	 2.0844007e-01
      60	 1.2520302e+00	 1.3450235e-01	 1.3025041e+00	 1.4566280e-01	[ 1.1779943e+00]	 1.9709325e-01


.. parsed-literal::

      61	 1.2623918e+00	 1.3405885e-01	 1.3124651e+00	 1.4538145e-01	[ 1.1869319e+00]	 1.8672442e-01
      62	 1.2694347e+00	 1.3366229e-01	 1.3194743e+00	 1.4507950e-01	[ 1.1890991e+00]	 1.6554832e-01


.. parsed-literal::

      63	 1.2767765e+00	 1.3339199e-01	 1.3268872e+00	 1.4480860e-01	[ 1.1908625e+00]	 1.7549253e-01
      64	 1.2884864e+00	 1.3261698e-01	 1.3387538e+00	 1.4412611e-01	[ 1.1932354e+00]	 1.9247055e-01


.. parsed-literal::

      65	 1.2969583e+00	 1.3294314e-01	 1.3478177e+00	 1.4427408e-01	[ 1.1944604e+00]	 1.7456365e-01


.. parsed-literal::

      66	 1.3048937e+00	 1.3191543e-01	 1.3554977e+00	 1.4349201e-01	[ 1.2028148e+00]	 2.1741819e-01


.. parsed-literal::

      67	 1.3097755e+00	 1.3145066e-01	 1.3605208e+00	 1.4303206e-01	[ 1.2051722e+00]	 2.1378565e-01


.. parsed-literal::

      68	 1.3176957e+00	 1.3097841e-01	 1.3688788e+00	 1.4247394e-01	[ 1.2071773e+00]	 2.1876454e-01
      69	 1.3259367e+00	 1.3056912e-01	 1.3775309e+00	 1.4193004e-01	[ 1.2181832e+00]	 1.9362569e-01


.. parsed-literal::

      70	 1.3358921e+00	 1.3027955e-01	 1.3875823e+00	 1.4170945e-01	[ 1.2268724e+00]	 2.1702647e-01


.. parsed-literal::

      71	 1.3437007e+00	 1.3002523e-01	 1.3954468e+00	 1.4131428e-01	[ 1.2342801e+00]	 2.1189904e-01


.. parsed-literal::

      72	 1.3525459e+00	 1.2976702e-01	 1.4044937e+00	 1.4100225e-01	[ 1.2414311e+00]	 2.2098851e-01


.. parsed-literal::

      73	 1.3586054e+00	 1.2965005e-01	 1.4108044e+00	 1.4032475e-01	  1.2400341e+00 	 3.3289957e-01


.. parsed-literal::

      74	 1.3650390e+00	 1.2952369e-01	 1.4172073e+00	 1.3971102e-01	[ 1.2465522e+00]	 2.1516991e-01


.. parsed-literal::

      75	 1.3708520e+00	 1.2958609e-01	 1.4232743e+00	 1.3911797e-01	[ 1.2508868e+00]	 2.1074677e-01


.. parsed-literal::

      76	 1.3761356e+00	 1.2960408e-01	 1.4285893e+00	 1.3869115e-01	[ 1.2533378e+00]	 2.1300793e-01


.. parsed-literal::

      77	 1.3813799e+00	 1.2934817e-01	 1.4339107e+00	 1.3808049e-01	  1.2522899e+00 	 2.2116399e-01


.. parsed-literal::

      78	 1.3871135e+00	 1.2939358e-01	 1.4400167e+00	 1.3787232e-01	  1.2532337e+00 	 2.1216488e-01
      79	 1.3925945e+00	 1.2873207e-01	 1.4455924e+00	 1.3739589e-01	  1.2496844e+00 	 1.9256949e-01


.. parsed-literal::

      80	 1.3979219e+00	 1.2851053e-01	 1.4508413e+00	 1.3733557e-01	[ 1.2533980e+00]	 2.1430826e-01
      81	 1.4037059e+00	 1.2818040e-01	 1.4567466e+00	 1.3786862e-01	[ 1.2538201e+00]	 2.0001292e-01


.. parsed-literal::

      82	 1.4066187e+00	 1.2850735e-01	 1.4597462e+00	 1.3838710e-01	  1.2532680e+00 	 2.1207857e-01


.. parsed-literal::

      83	 1.4097372e+00	 1.2825890e-01	 1.4628011e+00	 1.3841179e-01	[ 1.2572096e+00]	 2.1615863e-01
      84	 1.4136327e+00	 1.2792502e-01	 1.4668272e+00	 1.3857973e-01	  1.2567618e+00 	 1.9271135e-01


.. parsed-literal::

      85	 1.4168490e+00	 1.2776540e-01	 1.4700962e+00	 1.3870119e-01	  1.2566176e+00 	 2.2009492e-01


.. parsed-literal::

      86	 1.4251246e+00	 1.2746010e-01	 1.4784968e+00	 1.3929468e-01	  1.2508503e+00 	 2.1357942e-01


.. parsed-literal::

      87	 1.4286495e+00	 1.2744755e-01	 1.4820992e+00	 1.3948461e-01	  1.2528313e+00 	 3.3663273e-01


.. parsed-literal::

      88	 1.4317787e+00	 1.2738277e-01	 1.4851401e+00	 1.3950929e-01	  1.2537911e+00 	 2.1522069e-01


.. parsed-literal::

      89	 1.4346889e+00	 1.2730875e-01	 1.4880607e+00	 1.3973482e-01	  1.2534399e+00 	 2.2114897e-01


.. parsed-literal::

      90	 1.4378880e+00	 1.2723782e-01	 1.4913476e+00	 1.3987522e-01	  1.2473836e+00 	 2.2831368e-01


.. parsed-literal::

      91	 1.4415307e+00	 1.2712034e-01	 1.4950245e+00	 1.4005437e-01	  1.2467912e+00 	 2.1447039e-01


.. parsed-literal::

      92	 1.4455647e+00	 1.2700892e-01	 1.4991569e+00	 1.4037025e-01	  1.2438055e+00 	 2.1784091e-01
      93	 1.4487268e+00	 1.2689135e-01	 1.5023742e+00	 1.3990676e-01	  1.2426872e+00 	 1.8007565e-01


.. parsed-literal::

      94	 1.4520384e+00	 1.2675578e-01	 1.5055947e+00	 1.3949733e-01	  1.2432795e+00 	 2.1017599e-01
      95	 1.4551972e+00	 1.2665748e-01	 1.5087176e+00	 1.3897354e-01	  1.2433808e+00 	 1.9846439e-01


.. parsed-literal::

      96	 1.4579655e+00	 1.2636959e-01	 1.5114730e+00	 1.3833914e-01	  1.2428013e+00 	 2.0778227e-01
      97	 1.4612635e+00	 1.2634634e-01	 1.5148393e+00	 1.3803461e-01	  1.2404716e+00 	 1.7835760e-01


.. parsed-literal::

      98	 1.4639325e+00	 1.2615385e-01	 1.5175105e+00	 1.3799513e-01	  1.2392182e+00 	 2.0962143e-01


.. parsed-literal::

      99	 1.4667310e+00	 1.2592620e-01	 1.5204549e+00	 1.3820990e-01	  1.2278929e+00 	 2.1328878e-01


.. parsed-literal::

     100	 1.4691755e+00	 1.2571869e-01	 1.5228600e+00	 1.3841306e-01	  1.2263862e+00 	 2.2337413e-01


.. parsed-literal::

     101	 1.4711737e+00	 1.2565153e-01	 1.5248394e+00	 1.3854030e-01	  1.2233866e+00 	 2.1280622e-01


.. parsed-literal::

     102	 1.4759953e+00	 1.2549006e-01	 1.5297255e+00	 1.3905297e-01	  1.2102607e+00 	 2.1413517e-01


.. parsed-literal::

     103	 1.4783217e+00	 1.2546889e-01	 1.5321301e+00	 1.3912623e-01	  1.1935876e+00 	 2.1874833e-01


.. parsed-literal::

     104	 1.4819750e+00	 1.2529868e-01	 1.5357390e+00	 1.3934637e-01	  1.1923803e+00 	 2.2001100e-01


.. parsed-literal::

     105	 1.4841265e+00	 1.2514924e-01	 1.5378861e+00	 1.3934497e-01	  1.1913439e+00 	 2.0716429e-01


.. parsed-literal::

     106	 1.4861188e+00	 1.2501419e-01	 1.5399423e+00	 1.3923924e-01	  1.1844460e+00 	 2.1440434e-01
     107	 1.4880585e+00	 1.2463206e-01	 1.5420874e+00	 1.3909767e-01	  1.1554812e+00 	 1.9925404e-01


.. parsed-literal::

     108	 1.4906212e+00	 1.2459520e-01	 1.5446284e+00	 1.3889643e-01	  1.1495589e+00 	 2.1497941e-01


.. parsed-literal::

     109	 1.4922787e+00	 1.2445261e-01	 1.5463089e+00	 1.3874664e-01	  1.1391318e+00 	 2.0309496e-01
     110	 1.4941023e+00	 1.2429438e-01	 1.5481619e+00	 1.3858962e-01	  1.1244456e+00 	 1.9978619e-01


.. parsed-literal::

     111	 1.4952117e+00	 1.2389229e-01	 1.5493714e+00	 1.3860896e-01	  1.0971766e+00 	 2.1068549e-01


.. parsed-literal::

     112	 1.4978875e+00	 1.2385278e-01	 1.5519806e+00	 1.3847973e-01	  1.0973323e+00 	 2.2054911e-01


.. parsed-literal::

     113	 1.4991162e+00	 1.2379124e-01	 1.5531824e+00	 1.3846413e-01	  1.0988216e+00 	 2.1195936e-01


.. parsed-literal::

     114	 1.5008859e+00	 1.2364511e-01	 1.5549461e+00	 1.3843963e-01	  1.0989410e+00 	 2.1025062e-01


.. parsed-literal::

     115	 1.5037322e+00	 1.2338471e-01	 1.5578384e+00	 1.3843623e-01	  1.0958661e+00 	 2.1356940e-01


.. parsed-literal::

     116	 1.5052764e+00	 1.2325940e-01	 1.5594488e+00	 1.3852340e-01	  1.0954824e+00 	 3.3225107e-01


.. parsed-literal::

     117	 1.5077738e+00	 1.2308282e-01	 1.5620084e+00	 1.3859089e-01	  1.0921467e+00 	 2.0881557e-01
     118	 1.5100358e+00	 1.2300195e-01	 1.5643452e+00	 1.3861162e-01	  1.0887190e+00 	 1.8708014e-01


.. parsed-literal::

     119	 1.5127457e+00	 1.2302792e-01	 1.5671816e+00	 1.3849484e-01	  1.0856273e+00 	 2.0288610e-01


.. parsed-literal::

     120	 1.5141147e+00	 1.2299328e-01	 1.5686937e+00	 1.3821625e-01	  1.0897193e+00 	 2.1503186e-01


.. parsed-literal::

     121	 1.5161292e+00	 1.2299828e-01	 1.5705819e+00	 1.3798255e-01	  1.0950556e+00 	 2.2338486e-01
     122	 1.5173321e+00	 1.2297571e-01	 1.5717131e+00	 1.3769804e-01	  1.1020245e+00 	 1.9241214e-01


.. parsed-literal::

     123	 1.5187146e+00	 1.2291053e-01	 1.5730596e+00	 1.3735457e-01	  1.1085121e+00 	 2.1378946e-01


.. parsed-literal::

     124	 1.5214513e+00	 1.2281957e-01	 1.5758358e+00	 1.3676204e-01	  1.1183541e+00 	 2.0233035e-01


.. parsed-literal::

     125	 1.5223137e+00	 1.2256158e-01	 1.5768067e+00	 1.3630198e-01	  1.1236890e+00 	 2.1953678e-01
     126	 1.5245168e+00	 1.2254701e-01	 1.5789550e+00	 1.3635442e-01	  1.1216068e+00 	 1.9127774e-01


.. parsed-literal::

     127	 1.5252890e+00	 1.2246210e-01	 1.5797483e+00	 1.3640788e-01	  1.1193922e+00 	 2.1398902e-01


.. parsed-literal::

     128	 1.5269368e+00	 1.2225999e-01	 1.5814724e+00	 1.3633208e-01	  1.1184623e+00 	 2.1625686e-01


.. parsed-literal::

     129	 1.5275626e+00	 1.2181858e-01	 1.5822643e+00	 1.3648458e-01	  1.1154829e+00 	 2.2044373e-01


.. parsed-literal::

     130	 1.5299979e+00	 1.2182907e-01	 1.5845897e+00	 1.3621295e-01	  1.1237651e+00 	 2.1707630e-01


.. parsed-literal::

     131	 1.5310175e+00	 1.2175727e-01	 1.5855820e+00	 1.3600924e-01	  1.1292174e+00 	 2.1315622e-01


.. parsed-literal::

     132	 1.5321791e+00	 1.2163934e-01	 1.5867292e+00	 1.3586135e-01	  1.1334106e+00 	 2.1556711e-01


.. parsed-literal::

     133	 1.5341662e+00	 1.2144027e-01	 1.5887185e+00	 1.3564545e-01	  1.1335094e+00 	 2.1561241e-01


.. parsed-literal::

     134	 1.5353866e+00	 1.2138655e-01	 1.5899846e+00	 1.3565018e-01	  1.1330413e+00 	 3.0944324e-01


.. parsed-literal::

     135	 1.5372229e+00	 1.2124709e-01	 1.5918541e+00	 1.3545494e-01	  1.1231649e+00 	 2.1520138e-01
     136	 1.5384952e+00	 1.2113287e-01	 1.5931649e+00	 1.3532066e-01	  1.1138225e+00 	 1.9189882e-01


.. parsed-literal::

     137	 1.5397977e+00	 1.2111917e-01	 1.5945191e+00	 1.3510916e-01	  1.1013231e+00 	 2.1195221e-01


.. parsed-literal::

     138	 1.5415683e+00	 1.2102979e-01	 1.5963493e+00	 1.3482073e-01	  1.0857771e+00 	 2.0979977e-01
     139	 1.5425857e+00	 1.2096004e-01	 1.5974821e+00	 1.3404872e-01	  1.0759556e+00 	 1.8859696e-01


.. parsed-literal::

     140	 1.5440697e+00	 1.2086201e-01	 1.5988905e+00	 1.3408309e-01	  1.0760684e+00 	 2.2131133e-01


.. parsed-literal::

     141	 1.5449384e+00	 1.2076641e-01	 1.5997109e+00	 1.3403145e-01	  1.0780132e+00 	 2.1505284e-01


.. parsed-literal::

     142	 1.5467262e+00	 1.2061916e-01	 1.6014888e+00	 1.3367503e-01	  1.0702691e+00 	 2.2282243e-01


.. parsed-literal::

     143	 1.5481137e+00	 1.2054764e-01	 1.6029116e+00	 1.3351349e-01	  1.0571999e+00 	 3.1827569e-01
     144	 1.5497883e+00	 1.2053769e-01	 1.6046161e+00	 1.3315414e-01	  1.0397012e+00 	 2.0476675e-01


.. parsed-literal::

     145	 1.5512768e+00	 1.2054903e-01	 1.6061416e+00	 1.3298777e-01	  1.0197694e+00 	 1.8137956e-01


.. parsed-literal::

     146	 1.5527671e+00	 1.2064541e-01	 1.6076849e+00	 1.3292838e-01	  9.8954496e-01 	 2.1191645e-01


.. parsed-literal::

     147	 1.5540807e+00	 1.2062075e-01	 1.6089827e+00	 1.3303852e-01	  9.7691246e-01 	 2.1140862e-01


.. parsed-literal::

     148	 1.5555009e+00	 1.2054812e-01	 1.6103939e+00	 1.3315964e-01	  9.6406684e-01 	 2.1474743e-01
     149	 1.5563059e+00	 1.2049826e-01	 1.6111982e+00	 1.3326886e-01	  9.5717268e-01 	 1.8450737e-01


.. parsed-literal::

     150	 1.5573708e+00	 1.2043709e-01	 1.6122380e+00	 1.3316562e-01	  9.5437554e-01 	 2.1438670e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 10s, sys: 1.42 s, total: 2min 11s
    Wall time: 33.1 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fc69056cdf0>



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
    CPU times: user 1.88 s, sys: 68 ms, total: 1.94 s
    Wall time: 648 ms


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

