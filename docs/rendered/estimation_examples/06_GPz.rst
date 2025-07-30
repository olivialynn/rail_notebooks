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
       1	-3.4287380e-01	 3.2069870e-01	-3.3313248e-01	 3.2327667e-01	[-3.3427868e-01]	 4.6300578e-01


.. parsed-literal::

       2	-2.7537099e-01	 3.1115941e-01	-2.5238376e-01	 3.0996497e-01	[-2.4788332e-01]	 2.3590755e-01


.. parsed-literal::

       3	-2.2889133e-01	 2.8887462e-01	-1.8638065e-01	 2.9256175e-01	[-1.8901475e-01]	 2.8637815e-01
       4	-1.9455020e-01	 2.6577266e-01	-1.5094076e-01	 2.6978247e-01	[-1.6426562e-01]	 1.8422031e-01


.. parsed-literal::

       5	-1.0630696e-01	 2.5768470e-01	-7.3215140e-02	 2.6600393e-01	[-9.8954067e-02]	 2.1563387e-01


.. parsed-literal::

       6	-7.2020565e-02	 2.5183961e-01	-4.1514296e-02	 2.5731300e-01	[-5.9805700e-02]	 2.0685482e-01


.. parsed-literal::

       7	-5.4571206e-02	 2.4922847e-01	-2.9685620e-02	 2.5473725e-01	[-5.0043380e-02]	 2.1923971e-01
       8	-3.7328988e-02	 2.4615368e-01	-1.6887439e-02	 2.5145484e-01	[-3.7757828e-02]	 1.8829870e-01


.. parsed-literal::

       9	-2.2509972e-02	 2.4334392e-01	-4.9342324e-03	 2.4916766e-01	[-2.8167829e-02]	 2.1405172e-01
      10	-9.2981814e-03	 2.4085637e-01	 6.0140243e-03	 2.4649949e-01	[-1.6914254e-02]	 1.9839716e-01


.. parsed-literal::

      11	-4.9432667e-03	 2.4033224e-01	 8.8661168e-03	 2.4745688e-01	 -2.0148586e-02 	 1.9172692e-01


.. parsed-literal::

      12	-1.6144054e-03	 2.3968200e-01	 1.1942498e-02	 2.4651250e-01	[-1.5071962e-02]	 2.2044921e-01


.. parsed-literal::

      13	 8.3107072e-04	 2.3924651e-01	 1.4325695e-02	 2.4597123e-01	[-1.3072978e-02]	 2.1567416e-01
      14	 6.2246453e-03	 2.3815157e-01	 2.0190251e-02	 2.4499093e-01	[-7.7935506e-03]	 1.8173623e-01


.. parsed-literal::

      15	 9.0587209e-02	 2.2102485e-01	 1.1181564e-01	 2.3040364e-01	[ 9.6491178e-02]	 2.9180384e-01


.. parsed-literal::

      16	 2.0436350e-01	 2.1620515e-01	 2.2986683e-01	 2.2667661e-01	[ 2.0427416e-01]	 2.1376204e-01


.. parsed-literal::

      17	 2.8762791e-01	 2.1384920e-01	 3.1955441e-01	 2.2121314e-01	[ 2.7555596e-01]	 2.0466971e-01
      18	 3.5595654e-01	 2.1480621e-01	 3.8683177e-01	 2.2261912e-01	[ 3.5502766e-01]	 1.8796945e-01


.. parsed-literal::

      19	 3.9831419e-01	 2.1006736e-01	 4.3081522e-01	 2.1727108e-01	[ 4.0472228e-01]	 2.1333170e-01


.. parsed-literal::

      20	 4.5600939e-01	 2.0687851e-01	 4.8788636e-01	 2.1449452e-01	[ 4.6564975e-01]	 2.0435739e-01
      21	 5.1032591e-01	 2.0774436e-01	 5.4433870e-01	 2.1515010e-01	[ 5.1607685e-01]	 2.0015717e-01


.. parsed-literal::

      22	 5.5814277e-01	 2.0502339e-01	 5.9442399e-01	 2.1773446e-01	[ 5.5713403e-01]	 2.0518255e-01


.. parsed-literal::

      23	 5.9603896e-01	 2.0139559e-01	 6.3322988e-01	 2.1246105e-01	[ 6.0066643e-01]	 2.0696545e-01


.. parsed-literal::

      24	 6.3715746e-01	 1.9787960e-01	 6.7550774e-01	 2.0999007e-01	[ 6.4968634e-01]	 2.1552205e-01
      25	 6.6969658e-01	 1.9704579e-01	 7.0892581e-01	 2.0602150e-01	[ 6.8759285e-01]	 2.0236158e-01


.. parsed-literal::

      26	 7.0182791e-01	 1.9501664e-01	 7.4038803e-01	 2.0351070e-01	[ 7.2045835e-01]	 2.1718836e-01


.. parsed-literal::

      27	 7.3815100e-01	 1.9722382e-01	 7.7592136e-01	 2.0418689e-01	[ 7.6215655e-01]	 2.1078634e-01


.. parsed-literal::

      28	 7.5793920e-01	 1.9962000e-01	 7.9523122e-01	 2.0577261e-01	[ 7.9035545e-01]	 3.2317352e-01


.. parsed-literal::

      29	 7.7977906e-01	 1.9991369e-01	 8.1793153e-01	 2.0491060e-01	[ 8.1013415e-01]	 2.1340775e-01


.. parsed-literal::

      30	 8.0304822e-01	 1.9716619e-01	 8.4239086e-01	 2.0187321e-01	[ 8.2476667e-01]	 2.0208573e-01
      31	 8.3735098e-01	 1.9097283e-01	 8.7802151e-01	 1.9646157e-01	[ 8.5290664e-01]	 1.8381572e-01


.. parsed-literal::

      32	 8.5234730e-01	 1.8735261e-01	 8.9328917e-01	 1.9750530e-01	[ 8.5489784e-01]	 1.9710469e-01


.. parsed-literal::

      33	 8.7307545e-01	 1.8494091e-01	 9.1326868e-01	 1.9266047e-01	[ 8.8477015e-01]	 2.0849991e-01


.. parsed-literal::

      34	 8.8295268e-01	 1.8284699e-01	 9.2307001e-01	 1.8856652e-01	[ 8.9710694e-01]	 2.0914841e-01


.. parsed-literal::

      35	 8.9476454e-01	 1.8105723e-01	 9.3520521e-01	 1.8651223e-01	[ 9.0860403e-01]	 2.0845151e-01
      36	 9.1858396e-01	 1.7840978e-01	 9.6032636e-01	 1.8457399e-01	[ 9.2400981e-01]	 1.8174767e-01


.. parsed-literal::

      37	 9.3058814e-01	 1.7772373e-01	 9.7326697e-01	 1.8502011e-01	[ 9.2925044e-01]	 2.1090317e-01


.. parsed-literal::

      38	 9.4174909e-01	 1.7625132e-01	 9.8406677e-01	 1.8402257e-01	[ 9.4198185e-01]	 2.1046066e-01


.. parsed-literal::

      39	 9.5179654e-01	 1.7506289e-01	 9.9423976e-01	 1.8258753e-01	[ 9.5078385e-01]	 2.0691800e-01


.. parsed-literal::

      40	 9.6471221e-01	 1.7407997e-01	 1.0075606e+00	 1.8147558e-01	[ 9.6379606e-01]	 2.0638728e-01


.. parsed-literal::

      41	 9.8163519e-01	 1.7340687e-01	 1.0255060e+00	 1.8144268e-01	[ 9.7842432e-01]	 2.1515226e-01


.. parsed-literal::

      42	 9.9692453e-01	 1.7176237e-01	 1.0412680e+00	 1.8071298e-01	[ 9.9064930e-01]	 2.0556045e-01


.. parsed-literal::

      43	 1.0110358e+00	 1.6989048e-01	 1.0556408e+00	 1.7963224e-01	[ 1.0018821e+00]	 2.0321918e-01


.. parsed-literal::

      44	 1.0272536e+00	 1.6759925e-01	 1.0728295e+00	 1.7819290e-01	  9.9863834e-01 	 2.1465564e-01


.. parsed-literal::

      45	 1.0372100e+00	 1.6700954e-01	 1.0832558e+00	 1.7798477e-01	[ 1.0144225e+00]	 2.0111132e-01


.. parsed-literal::

      46	 1.0445453e+00	 1.6673564e-01	 1.0900616e+00	 1.7694867e-01	[ 1.0203335e+00]	 2.1350431e-01


.. parsed-literal::

      47	 1.0526433e+00	 1.6624379e-01	 1.0983341e+00	 1.7630209e-01	[ 1.0242557e+00]	 2.1664405e-01


.. parsed-literal::

      48	 1.0634092e+00	 1.6532854e-01	 1.1094890e+00	 1.7537877e-01	[ 1.0293962e+00]	 2.0967412e-01


.. parsed-literal::

      49	 1.0770418e+00	 1.6255126e-01	 1.1234462e+00	 1.7425526e-01	[ 1.0389419e+00]	 2.1800685e-01
      50	 1.0860456e+00	 1.6017743e-01	 1.1328961e+00	 1.7337016e-01	[ 1.0442561e+00]	 1.9820833e-01


.. parsed-literal::

      51	 1.0928246e+00	 1.5952293e-01	 1.1394942e+00	 1.7265324e-01	[ 1.0510248e+00]	 2.0382643e-01


.. parsed-literal::

      52	 1.1023765e+00	 1.5827673e-01	 1.1491336e+00	 1.7115282e-01	[ 1.0576122e+00]	 2.0511794e-01


.. parsed-literal::

      53	 1.1106342e+00	 1.5679446e-01	 1.1576433e+00	 1.6951266e-01	[ 1.0655444e+00]	 2.1767044e-01


.. parsed-literal::

      54	 1.1195415e+00	 1.5301723e-01	 1.1674077e+00	 1.6574438e-01	[ 1.0725528e+00]	 2.0901179e-01


.. parsed-literal::

      55	 1.1316776e+00	 1.5117829e-01	 1.1793323e+00	 1.6429518e-01	[ 1.0875510e+00]	 2.0427060e-01


.. parsed-literal::

      56	 1.1369867e+00	 1.5052248e-01	 1.1843848e+00	 1.6376813e-01	[ 1.0968743e+00]	 2.1180582e-01
      57	 1.1448505e+00	 1.4817181e-01	 1.1923235e+00	 1.6161842e-01	[ 1.1034920e+00]	 1.8673182e-01


.. parsed-literal::

      58	 1.1524361e+00	 1.4664918e-01	 1.2000616e+00	 1.5978658e-01	[ 1.1152665e+00]	 2.0755529e-01
      59	 1.1579998e+00	 1.4538059e-01	 1.2058092e+00	 1.5804063e-01	[ 1.1205838e+00]	 1.9756484e-01


.. parsed-literal::

      60	 1.1682799e+00	 1.4333609e-01	 1.2163182e+00	 1.5470233e-01	[ 1.1300815e+00]	 2.1619487e-01


.. parsed-literal::

      61	 1.1740611e+00	 1.4199827e-01	 1.2225087e+00	 1.5330287e-01	  1.1282000e+00 	 2.1229458e-01


.. parsed-literal::

      62	 1.1832946e+00	 1.4159166e-01	 1.2315298e+00	 1.5253969e-01	[ 1.1379794e+00]	 2.1911335e-01
      63	 1.1890559e+00	 1.4120996e-01	 1.2372704e+00	 1.5235913e-01	[ 1.1421931e+00]	 1.9123387e-01


.. parsed-literal::

      64	 1.1960032e+00	 1.4101953e-01	 1.2443050e+00	 1.5269094e-01	[ 1.1460568e+00]	 2.0281386e-01
      65	 1.2056275e+00	 1.3959632e-01	 1.2545264e+00	 1.5194757e-01	[ 1.1511904e+00]	 1.8462467e-01


.. parsed-literal::

      66	 1.2161374e+00	 1.3932400e-01	 1.2654314e+00	 1.5249506e-01	[ 1.1564737e+00]	 2.1023226e-01


.. parsed-literal::

      67	 1.2230013e+00	 1.3859756e-01	 1.2724893e+00	 1.5170032e-01	[ 1.1636903e+00]	 2.1666598e-01


.. parsed-literal::

      68	 1.2353867e+00	 1.3744214e-01	 1.2857792e+00	 1.5031275e-01	[ 1.1736262e+00]	 2.1137953e-01


.. parsed-literal::

      69	 1.2393885e+00	 1.3711656e-01	 1.2901530e+00	 1.4982931e-01	[ 1.1822491e+00]	 2.0429850e-01
      70	 1.2473994e+00	 1.3693658e-01	 1.2978697e+00	 1.5016733e-01	[ 1.1861809e+00]	 1.8184853e-01


.. parsed-literal::

      71	 1.2532983e+00	 1.3677564e-01	 1.3038690e+00	 1.5087761e-01	[ 1.1894388e+00]	 2.0174479e-01


.. parsed-literal::

      72	 1.2577018e+00	 1.3653279e-01	 1.3083625e+00	 1.5141246e-01	[ 1.1946589e+00]	 2.0907831e-01


.. parsed-literal::

      73	 1.2610907e+00	 1.3623659e-01	 1.3119793e+00	 1.5188520e-01	  1.1932212e+00 	 2.9146218e-01
      74	 1.2670347e+00	 1.3564349e-01	 1.3179974e+00	 1.5217132e-01	[ 1.2020697e+00]	 1.9674277e-01


.. parsed-literal::

      75	 1.2726998e+00	 1.3482466e-01	 1.3238108e+00	 1.5222017e-01	[ 1.2084109e+00]	 2.0024610e-01


.. parsed-literal::

      76	 1.2807879e+00	 1.3389947e-01	 1.3318862e+00	 1.5231750e-01	[ 1.2152199e+00]	 2.0835567e-01
      77	 1.2854921e+00	 1.3268384e-01	 1.3366365e+00	 1.5202351e-01	[ 1.2270830e+00]	 1.7556357e-01


.. parsed-literal::

      78	 1.2920070e+00	 1.3276575e-01	 1.3428914e+00	 1.5239357e-01	[ 1.2292388e+00]	 2.0654845e-01


.. parsed-literal::

      79	 1.2953413e+00	 1.3278013e-01	 1.3461464e+00	 1.5242734e-01	[ 1.2299321e+00]	 2.0791149e-01
      80	 1.3025575e+00	 1.3258782e-01	 1.3533816e+00	 1.5250929e-01	[ 1.2336382e+00]	 1.8670368e-01


.. parsed-literal::

      81	 1.3087085e+00	 1.3221143e-01	 1.3597181e+00	 1.5246489e-01	[ 1.2363578e+00]	 1.8625808e-01


.. parsed-literal::

      82	 1.3144903e+00	 1.3187497e-01	 1.3657809e+00	 1.5230210e-01	[ 1.2464080e+00]	 2.2124791e-01


.. parsed-literal::

      83	 1.3198831e+00	 1.3154385e-01	 1.3712505e+00	 1.5225476e-01	[ 1.2507673e+00]	 2.0386410e-01
      84	 1.3250094e+00	 1.3126166e-01	 1.3764902e+00	 1.5233393e-01	[ 1.2545924e+00]	 1.9667721e-01


.. parsed-literal::

      85	 1.3278015e+00	 1.3109825e-01	 1.3793286e+00	 1.5171028e-01	[ 1.2586592e+00]	 3.3400893e-01


.. parsed-literal::

      86	 1.3314025e+00	 1.3104916e-01	 1.3829998e+00	 1.5163665e-01	  1.2586308e+00 	 2.1007395e-01


.. parsed-literal::

      87	 1.3360186e+00	 1.3109123e-01	 1.3877339e+00	 1.5136354e-01	  1.2571346e+00 	 2.1707535e-01


.. parsed-literal::

      88	 1.3392485e+00	 1.3099352e-01	 1.3910304e+00	 1.5093651e-01	  1.2572749e+00 	 2.0472455e-01


.. parsed-literal::

      89	 1.3462760e+00	 1.3066298e-01	 1.3982226e+00	 1.4995215e-01	[ 1.2611975e+00]	 2.0514703e-01


.. parsed-literal::

      90	 1.3504044e+00	 1.3001083e-01	 1.4025508e+00	 1.4840067e-01	[ 1.2670394e+00]	 3.2245421e-01


.. parsed-literal::

      91	 1.3551521e+00	 1.2984338e-01	 1.4073348e+00	 1.4798896e-01	[ 1.2701406e+00]	 2.0332575e-01


.. parsed-literal::

      92	 1.3596345e+00	 1.2949978e-01	 1.4118545e+00	 1.4726192e-01	[ 1.2738931e+00]	 2.0742321e-01


.. parsed-literal::

      93	 1.3656003e+00	 1.2949996e-01	 1.4179554e+00	 1.4671561e-01	[ 1.2740368e+00]	 2.0549250e-01


.. parsed-literal::

      94	 1.3708932e+00	 1.2938547e-01	 1.4232313e+00	 1.4580646e-01	[ 1.2782537e+00]	 2.1748662e-01


.. parsed-literal::

      95	 1.3753916e+00	 1.2945189e-01	 1.4276660e+00	 1.4556542e-01	[ 1.2820620e+00]	 2.1468115e-01


.. parsed-literal::

      96	 1.3800840e+00	 1.2953918e-01	 1.4324016e+00	 1.4540906e-01	[ 1.2871807e+00]	 2.1207595e-01
      97	 1.3833373e+00	 1.2972952e-01	 1.4358303e+00	 1.4547760e-01	[ 1.2875566e+00]	 1.8226743e-01


.. parsed-literal::

      98	 1.3869766e+00	 1.2955715e-01	 1.4395165e+00	 1.4548801e-01	  1.2873601e+00 	 2.1166039e-01


.. parsed-literal::

      99	 1.3897373e+00	 1.2938836e-01	 1.4422733e+00	 1.4535244e-01	[ 1.2897940e+00]	 2.1721053e-01


.. parsed-literal::

     100	 1.3929492e+00	 1.2944477e-01	 1.4456379e+00	 1.4499763e-01	  1.2888310e+00 	 2.1937752e-01


.. parsed-literal::

     101	 1.3965835e+00	 1.2963509e-01	 1.4492988e+00	 1.4518791e-01	  1.2822038e+00 	 2.0866942e-01
     102	 1.4008609e+00	 1.2950656e-01	 1.4535333e+00	 1.4476393e-01	  1.2885128e+00 	 1.8753576e-01


.. parsed-literal::

     103	 1.4036263e+00	 1.2950983e-01	 1.4562865e+00	 1.4462375e-01	[ 1.2901526e+00]	 2.0407033e-01


.. parsed-literal::

     104	 1.4063627e+00	 1.2945056e-01	 1.4590253e+00	 1.4461079e-01	[ 1.2913506e+00]	 2.0802617e-01


.. parsed-literal::

     105	 1.4113633e+00	 1.2954225e-01	 1.4641948e+00	 1.4453967e-01	[ 1.2922657e+00]	 2.1892571e-01
     106	 1.4148378e+00	 1.3030184e-01	 1.4679623e+00	 1.4493077e-01	  1.2914257e+00 	 1.7961574e-01


.. parsed-literal::

     107	 1.4193258e+00	 1.3003101e-01	 1.4723139e+00	 1.4448276e-01	[ 1.2963269e+00]	 2.0496702e-01


.. parsed-literal::

     108	 1.4215527e+00	 1.3015743e-01	 1.4745040e+00	 1.4432279e-01	[ 1.2987008e+00]	 2.0803833e-01


.. parsed-literal::

     109	 1.4242924e+00	 1.3034481e-01	 1.4772910e+00	 1.4393677e-01	[ 1.3016679e+00]	 2.0766592e-01


.. parsed-literal::

     110	 1.4270602e+00	 1.3102058e-01	 1.4801489e+00	 1.4388081e-01	  1.2991766e+00 	 2.1866179e-01


.. parsed-literal::

     111	 1.4289040e+00	 1.3096790e-01	 1.4819799e+00	 1.4381101e-01	  1.2997309e+00 	 2.0703959e-01


.. parsed-literal::

     112	 1.4327912e+00	 1.3069704e-01	 1.4859697e+00	 1.4345566e-01	  1.2986830e+00 	 2.0265603e-01
     113	 1.4360477e+00	 1.3041975e-01	 1.4892904e+00	 1.4307249e-01	  1.2976171e+00 	 2.0469356e-01


.. parsed-literal::

     114	 1.4396061e+00	 1.3011925e-01	 1.4930591e+00	 1.4269449e-01	  1.2872503e+00 	 2.0310616e-01


.. parsed-literal::

     115	 1.4428463e+00	 1.3008928e-01	 1.4963117e+00	 1.4236309e-01	  1.2885447e+00 	 2.1960592e-01


.. parsed-literal::

     116	 1.4446948e+00	 1.3000018e-01	 1.4980664e+00	 1.4242439e-01	  1.2894236e+00 	 2.3820615e-01
     117	 1.4474828e+00	 1.2996314e-01	 1.5008907e+00	 1.4239871e-01	  1.2876030e+00 	 1.9246817e-01


.. parsed-literal::

     118	 1.4497382e+00	 1.3009637e-01	 1.5033954e+00	 1.4273154e-01	  1.2702065e+00 	 2.1294332e-01


.. parsed-literal::

     119	 1.4524406e+00	 1.2993204e-01	 1.5060773e+00	 1.4245941e-01	  1.2735945e+00 	 2.0614362e-01


.. parsed-literal::

     120	 1.4545277e+00	 1.2998307e-01	 1.5082131e+00	 1.4236489e-01	  1.2723874e+00 	 2.1093559e-01


.. parsed-literal::

     121	 1.4567432e+00	 1.3007995e-01	 1.5105344e+00	 1.4231546e-01	  1.2697245e+00 	 2.1341991e-01


.. parsed-literal::

     122	 1.4583715e+00	 1.3038069e-01	 1.5122643e+00	 1.4243428e-01	  1.2605213e+00 	 3.1701422e-01
     123	 1.4602901e+00	 1.3047554e-01	 1.5142377e+00	 1.4245244e-01	  1.2569282e+00 	 1.8386078e-01


.. parsed-literal::

     124	 1.4620131e+00	 1.3054565e-01	 1.5159778e+00	 1.4247187e-01	  1.2527297e+00 	 2.0762873e-01


.. parsed-literal::

     125	 1.4643626e+00	 1.3063199e-01	 1.5183635e+00	 1.4253931e-01	  1.2458590e+00 	 2.1838498e-01


.. parsed-literal::

     126	 1.4679779e+00	 1.3087724e-01	 1.5221334e+00	 1.4259774e-01	  1.2338752e+00 	 2.1310258e-01


.. parsed-literal::

     127	 1.4710777e+00	 1.3128070e-01	 1.5253578e+00	 1.4281254e-01	  1.2245763e+00 	 2.1443343e-01


.. parsed-literal::

     128	 1.4735934e+00	 1.3098471e-01	 1.5277471e+00	 1.4267640e-01	  1.2312189e+00 	 2.1602106e-01


.. parsed-literal::

     129	 1.4764137e+00	 1.3070505e-01	 1.5305336e+00	 1.4254269e-01	  1.2403067e+00 	 2.1262503e-01
     130	 1.4784277e+00	 1.3067303e-01	 1.5326073e+00	 1.4252574e-01	  1.2361947e+00 	 1.9839978e-01


.. parsed-literal::

     131	 1.4809976e+00	 1.3058932e-01	 1.5352014e+00	 1.4230272e-01	  1.2425797e+00 	 2.1366549e-01


.. parsed-literal::

     132	 1.4842102e+00	 1.3076711e-01	 1.5385286e+00	 1.4211958e-01	  1.2387053e+00 	 2.0541835e-01
     133	 1.4862963e+00	 1.3073331e-01	 1.5406772e+00	 1.4204760e-01	  1.2381217e+00 	 1.7974401e-01


.. parsed-literal::

     134	 1.4886637e+00	 1.3084617e-01	 1.5430283e+00	 1.4199971e-01	  1.2351704e+00 	 2.1250057e-01
     135	 1.4916054e+00	 1.3078855e-01	 1.5460031e+00	 1.4207607e-01	  1.2248967e+00 	 1.9703007e-01


.. parsed-literal::

     136	 1.4939957e+00	 1.3051939e-01	 1.5484693e+00	 1.4218944e-01	  1.2194293e+00 	 1.9669390e-01


.. parsed-literal::

     137	 1.4951380e+00	 1.3078818e-01	 1.5498741e+00	 1.4291533e-01	  1.2001980e+00 	 2.1277738e-01


.. parsed-literal::

     138	 1.4974539e+00	 1.3045248e-01	 1.5520364e+00	 1.4269467e-01	  1.2103901e+00 	 2.0403481e-01


.. parsed-literal::

     139	 1.4984217e+00	 1.3042669e-01	 1.5529929e+00	 1.4270619e-01	  1.2137712e+00 	 2.0609522e-01


.. parsed-literal::

     140	 1.5002985e+00	 1.3037875e-01	 1.5549124e+00	 1.4290826e-01	  1.2134675e+00 	 2.0781922e-01


.. parsed-literal::

     141	 1.5028079e+00	 1.3031213e-01	 1.5574923e+00	 1.4320761e-01	  1.2103801e+00 	 2.0216608e-01
     142	 1.5040566e+00	 1.2992684e-01	 1.5590544e+00	 1.4436075e-01	  1.1851251e+00 	 1.8902183e-01


.. parsed-literal::

     143	 1.5070684e+00	 1.2997407e-01	 1.5618995e+00	 1.4417826e-01	  1.1930954e+00 	 2.1611929e-01


.. parsed-literal::

     144	 1.5080889e+00	 1.2993592e-01	 1.5628566e+00	 1.4409982e-01	  1.1936456e+00 	 2.0935750e-01


.. parsed-literal::

     145	 1.5097463e+00	 1.2983066e-01	 1.5645316e+00	 1.4440546e-01	  1.1880416e+00 	 2.0923591e-01
     146	 1.5115244e+00	 1.2979706e-01	 1.5664275e+00	 1.4508952e-01	  1.1684405e+00 	 1.8892288e-01


.. parsed-literal::

     147	 1.5133703e+00	 1.2965854e-01	 1.5682918e+00	 1.4535652e-01	  1.1652198e+00 	 2.0418859e-01


.. parsed-literal::

     148	 1.5146066e+00	 1.2960917e-01	 1.5695613e+00	 1.4550026e-01	  1.1618318e+00 	 2.0944047e-01


.. parsed-literal::

     149	 1.5164972e+00	 1.2952857e-01	 1.5715364e+00	 1.4564481e-01	  1.1490905e+00 	 2.1882534e-01


.. parsed-literal::

     150	 1.5178940e+00	 1.2930362e-01	 1.5730856e+00	 1.4605515e-01	  1.1337701e+00 	 2.0238423e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 6s, sys: 1.24 s, total: 2min 7s
    Wall time: 32.1 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f88e06b45e0>



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
    CPU times: user 1.74 s, sys: 57 ms, total: 1.8 s
    Wall time: 562 ms


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

