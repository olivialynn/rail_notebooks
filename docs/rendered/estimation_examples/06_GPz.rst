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

    Inserting handle into data store.  input: None, GPz_Train
    ngal: 10225
    training model...


.. parsed-literal::

    Iter	 logML/n 		 Train RMSE		 Train RMSE/n		 Valid RMSE		 Valid MLL		 Time    
       1	-3.4985326e-01	 3.2240371e-01	-3.4020735e-01	 3.1345249e-01	[-3.2414566e-01]	 4.6442485e-01


.. parsed-literal::

       2	-2.7826158e-01	 3.1175254e-01	-2.5486370e-01	 3.0210272e-01	[-2.2699854e-01]	 2.3114443e-01


.. parsed-literal::

       3	-2.3401577e-01	 2.9147837e-01	-1.9317259e-01	 2.7922754e-01	[-1.4557491e-01]	 3.0050516e-01
       4	-2.0161714e-01	 2.6766277e-01	-1.6238098e-01	 2.5581070e-01	[-8.8952111e-02]	 1.9716763e-01


.. parsed-literal::

       5	-1.1199301e-01	 2.5924243e-01	-7.7873833e-02	 2.4655146e-01	[-2.0780895e-02]	 2.0590401e-01


.. parsed-literal::

       6	-7.8798606e-02	 2.5371754e-01	-4.8151908e-02	 2.4383992e-01	[-9.1966742e-03]	 2.0370936e-01


.. parsed-literal::

       7	-6.1136777e-02	 2.5109694e-01	-3.7327217e-02	 2.3996567e-01	[ 5.9896578e-03]	 2.0805550e-01
       8	-4.8896499e-02	 2.4904258e-01	-2.8821081e-02	 2.3769925e-01	[ 1.5380629e-02]	 1.8655252e-01


.. parsed-literal::

       9	-3.5001959e-02	 2.4645755e-01	-1.7972663e-02	 2.3456408e-01	[ 2.9982563e-02]	 1.7899275e-01


.. parsed-literal::

      10	-2.7482379e-02	 2.4541517e-01	-1.2737536e-02	 2.3368640e-01	[ 3.5193737e-02]	 2.1803784e-01


.. parsed-literal::

      11	-2.0828237e-02	 2.4383598e-01	-6.4951756e-03	 2.3186996e-01	[ 4.3711698e-02]	 2.1962333e-01


.. parsed-literal::

      12	-1.8297479e-02	 2.4334828e-01	-4.1304026e-03	 2.3111801e-01	[ 4.6862701e-02]	 2.1980858e-01


.. parsed-literal::

      13	-1.3456356e-02	 2.4235472e-01	 9.2780845e-04	 2.2994809e-01	[ 5.1582831e-02]	 2.0918512e-01


.. parsed-literal::

      14	 7.4204644e-02	 2.3070264e-01	 9.6396921e-02	 2.1983470e-01	[ 1.3773749e-01]	 3.2767177e-01


.. parsed-literal::

      15	 1.4637966e-01	 2.2354610e-01	 1.7041865e-01	 2.1600563e-01	[ 1.8898170e-01]	 3.3295488e-01


.. parsed-literal::

      16	 2.1503440e-01	 2.2514034e-01	 2.4912466e-01	 2.2220063e-01	[ 2.5766825e-01]	 2.1711087e-01


.. parsed-literal::

      17	 2.8963997e-01	 2.1454408e-01	 3.2316751e-01	 2.1282868e-01	[ 3.1861448e-01]	 2.0213246e-01


.. parsed-literal::

      18	 3.2022735e-01	 2.1093603e-01	 3.5259446e-01	 2.0702032e-01	[ 3.4263969e-01]	 2.0548034e-01


.. parsed-literal::

      19	 3.5807020e-01	 2.0633737e-01	 3.9040957e-01	 2.0121534e-01	[ 3.8169814e-01]	 2.0304418e-01
      20	 4.3645611e-01	 2.0231166e-01	 4.6994367e-01	 1.9848851e-01	[ 4.5642106e-01]	 2.0192528e-01


.. parsed-literal::

      21	 5.5391393e-01	 1.9673369e-01	 5.9003148e-01	 1.9192924e-01	[ 5.7090288e-01]	 2.0933175e-01
      22	 5.9549954e-01	 1.9988202e-01	 6.3452085e-01	 1.9531924e-01	[ 6.1728578e-01]	 1.9239902e-01


.. parsed-literal::

      23	 6.5419740e-01	 1.9391914e-01	 6.9268693e-01	 1.8638495e-01	[ 6.8079754e-01]	 2.1640563e-01


.. parsed-literal::

      24	 6.8766639e-01	 1.9233226e-01	 7.2636420e-01	 1.8550831e-01	[ 7.0529158e-01]	 2.1900964e-01
      25	 7.2388054e-01	 1.9109873e-01	 7.6169175e-01	 1.8590988e-01	[ 7.3604582e-01]	 1.9842958e-01


.. parsed-literal::

      26	 7.6624566e-01	 2.0095746e-01	 8.0475914e-01	 1.9242652e-01	[ 7.8464737e-01]	 2.0698357e-01


.. parsed-literal::

      27	 8.0428624e-01	 1.9755409e-01	 8.4271947e-01	 1.9400617e-01	[ 8.2805929e-01]	 2.0613027e-01
      28	 8.4811375e-01	 1.9170279e-01	 8.8820217e-01	 1.8777052e-01	[ 8.6129796e-01]	 1.9935107e-01


.. parsed-literal::

      29	 8.7344894e-01	 1.9117683e-01	 9.1499643e-01	 1.8510622e-01	[ 8.8317723e-01]	 2.1281219e-01
      30	 9.0312540e-01	 1.8830547e-01	 9.4623723e-01	 1.8370759e-01	[ 9.1153456e-01]	 1.8385482e-01


.. parsed-literal::

      31	 9.1937793e-01	 1.8706204e-01	 9.6231321e-01	 1.8346094e-01	[ 9.4231099e-01]	 1.7744994e-01


.. parsed-literal::

      32	 9.4496430e-01	 1.8278458e-01	 9.8875336e-01	 1.8014427e-01	[ 9.6275519e-01]	 2.0960307e-01
      33	 9.6685647e-01	 1.7969512e-01	 1.0113850e+00	 1.7754652e-01	[ 9.7713032e-01]	 1.8117428e-01


.. parsed-literal::

      34	 9.9549576e-01	 1.7627100e-01	 1.0409179e+00	 1.7462299e-01	[ 9.9581893e-01]	 1.7511010e-01
      35	 1.0082845e+00	 1.7010486e-01	 1.0551296e+00	 1.6789366e-01	[ 1.0019460e+00]	 1.8444586e-01


.. parsed-literal::

      36	 1.0286442e+00	 1.6932119e-01	 1.0741440e+00	 1.6705649e-01	[ 1.0295811e+00]	 2.0623589e-01
      37	 1.0409133e+00	 1.6867156e-01	 1.0863411e+00	 1.6644562e-01	[ 1.0402839e+00]	 1.8023610e-01


.. parsed-literal::

      38	 1.0560851e+00	 1.6560185e-01	 1.1015628e+00	 1.6328975e-01	[ 1.0563511e+00]	 2.1346092e-01


.. parsed-literal::

      39	 1.0718296e+00	 1.6189318e-01	 1.1180853e+00	 1.5942390e-01	[ 1.0656876e+00]	 2.0402956e-01


.. parsed-literal::

      40	 1.0872705e+00	 1.5876687e-01	 1.1340305e+00	 1.5697845e-01	[ 1.0851968e+00]	 2.2129154e-01


.. parsed-literal::

      41	 1.1035305e+00	 1.5585189e-01	 1.1506350e+00	 1.5398420e-01	[ 1.0989100e+00]	 2.0549917e-01


.. parsed-literal::

      42	 1.1225137e+00	 1.5296932e-01	 1.1700150e+00	 1.5111709e-01	[ 1.1146347e+00]	 2.2535038e-01
      43	 1.1383327e+00	 1.5144855e-01	 1.1859363e+00	 1.4903120e-01	[ 1.1402499e+00]	 1.8066144e-01


.. parsed-literal::

      44	 1.1516440e+00	 1.5088503e-01	 1.1993314e+00	 1.4795389e-01	[ 1.1486202e+00]	 2.1136761e-01


.. parsed-literal::

      45	 1.1670103e+00	 1.5011300e-01	 1.2148217e+00	 1.4662306e-01	[ 1.1574547e+00]	 2.0728636e-01
      46	 1.1834237e+00	 1.4850033e-01	 1.2313227e+00	 1.4448829e-01	[ 1.1740457e+00]	 1.9863153e-01


.. parsed-literal::

      47	 1.1965644e+00	 1.4698416e-01	 1.2444938e+00	 1.4343076e-01	[ 1.1835114e+00]	 2.0838213e-01


.. parsed-literal::

      48	 1.2068534e+00	 1.4600773e-01	 1.2549403e+00	 1.4254640e-01	[ 1.1948042e+00]	 2.1212053e-01
      49	 1.2199629e+00	 1.4508351e-01	 1.2682591e+00	 1.4196111e-01	[ 1.2062163e+00]	 1.8110943e-01


.. parsed-literal::

      50	 1.2324967e+00	 1.4290034e-01	 1.2810681e+00	 1.4051852e-01	[ 1.2137943e+00]	 2.1051502e-01


.. parsed-literal::

      51	 1.2476838e+00	 1.4100029e-01	 1.2964074e+00	 1.3940183e-01	[ 1.2204291e+00]	 2.1415448e-01
      52	 1.2592092e+00	 1.4009531e-01	 1.3081363e+00	 1.3893224e-01	[ 1.2299767e+00]	 1.9185138e-01


.. parsed-literal::

      53	 1.2683596e+00	 1.3974085e-01	 1.3175937e+00	 1.3893844e-01	[ 1.2328188e+00]	 2.0943260e-01


.. parsed-literal::

      54	 1.2792918e+00	 1.3883216e-01	 1.3288974e+00	 1.3842282e-01	[ 1.2378225e+00]	 2.1428442e-01


.. parsed-literal::

      55	 1.2892409e+00	 1.3728599e-01	 1.3388403e+00	 1.3724272e-01	[ 1.2461098e+00]	 2.2187757e-01
      56	 1.2986982e+00	 1.3681279e-01	 1.3484403e+00	 1.3663515e-01	[ 1.2585794e+00]	 1.8980551e-01


.. parsed-literal::

      57	 1.3088528e+00	 1.3551428e-01	 1.3590093e+00	 1.3502569e-01	[ 1.2679285e+00]	 1.9986606e-01
      58	 1.3195521e+00	 1.3443933e-01	 1.3693881e+00	 1.3331127e-01	[ 1.2909159e+00]	 1.8760037e-01


.. parsed-literal::

      59	 1.3286936e+00	 1.3366020e-01	 1.3792433e+00	 1.3224516e-01	[ 1.2953220e+00]	 1.9736958e-01


.. parsed-literal::

      60	 1.3352956e+00	 1.3307842e-01	 1.3862279e+00	 1.3182563e-01	  1.2936751e+00 	 2.0680475e-01


.. parsed-literal::

      61	 1.3414710e+00	 1.3271427e-01	 1.3924449e+00	 1.3149809e-01	[ 1.2987560e+00]	 2.1708083e-01


.. parsed-literal::

      62	 1.3480321e+00	 1.3189156e-01	 1.3989320e+00	 1.3126288e-01	[ 1.3046157e+00]	 2.0272541e-01


.. parsed-literal::

      63	 1.3563723e+00	 1.3106107e-01	 1.4073413e+00	 1.3094959e-01	[ 1.3124794e+00]	 2.1635294e-01
      64	 1.3642095e+00	 1.3013243e-01	 1.4154491e+00	 1.3075597e-01	[ 1.3170728e+00]	 1.9358921e-01


.. parsed-literal::

      65	 1.3719904e+00	 1.2939585e-01	 1.4233119e+00	 1.2964470e-01	[ 1.3236450e+00]	 1.7908192e-01


.. parsed-literal::

      66	 1.3775939e+00	 1.2917528e-01	 1.4288385e+00	 1.2898673e-01	[ 1.3308864e+00]	 2.1393681e-01


.. parsed-literal::

      67	 1.3847245e+00	 1.2859631e-01	 1.4362337e+00	 1.2790253e-01	[ 1.3329447e+00]	 2.1375918e-01


.. parsed-literal::

      68	 1.3899367e+00	 1.2818265e-01	 1.4416640e+00	 1.2729695e-01	[ 1.3347759e+00]	 2.0955348e-01


.. parsed-literal::

      69	 1.3952886e+00	 1.2806125e-01	 1.4467491e+00	 1.2731241e-01	[ 1.3417699e+00]	 2.0961785e-01


.. parsed-literal::

      70	 1.3991831e+00	 1.2790278e-01	 1.4506100e+00	 1.2728153e-01	[ 1.3435128e+00]	 2.0645428e-01


.. parsed-literal::

      71	 1.4053630e+00	 1.2757772e-01	 1.4568612e+00	 1.2683428e-01	[ 1.3480654e+00]	 2.1256804e-01


.. parsed-literal::

      72	 1.4095240e+00	 1.2702522e-01	 1.4615358e+00	 1.2632571e-01	  1.3420550e+00 	 2.1650124e-01


.. parsed-literal::

      73	 1.4171158e+00	 1.2675565e-01	 1.4690865e+00	 1.2550784e-01	[ 1.3539967e+00]	 2.0486617e-01


.. parsed-literal::

      74	 1.4211034e+00	 1.2675047e-01	 1.4731930e+00	 1.2516761e-01	[ 1.3577524e+00]	 2.0737433e-01
      75	 1.4254973e+00	 1.2679527e-01	 1.4777727e+00	 1.2497160e-01	[ 1.3620451e+00]	 1.8056107e-01


.. parsed-literal::

      76	 1.4294143e+00	 1.2680752e-01	 1.4818601e+00	 1.2482722e-01	[ 1.3633797e+00]	 2.1125174e-01


.. parsed-literal::

      77	 1.4332952e+00	 1.2683650e-01	 1.4856405e+00	 1.2507025e-01	[ 1.3704330e+00]	 2.1197844e-01


.. parsed-literal::

      78	 1.4362975e+00	 1.2682460e-01	 1.4885867e+00	 1.2529943e-01	[ 1.3724599e+00]	 2.2328472e-01


.. parsed-literal::

      79	 1.4393717e+00	 1.2680525e-01	 1.4916409e+00	 1.2564548e-01	[ 1.3735258e+00]	 2.1164632e-01


.. parsed-literal::

      80	 1.4415989e+00	 1.2697590e-01	 1.4939450e+00	 1.2618321e-01	  1.3707944e+00 	 2.1029997e-01


.. parsed-literal::

      81	 1.4445455e+00	 1.2683382e-01	 1.4968267e+00	 1.2597836e-01	[ 1.3748202e+00]	 2.2067642e-01


.. parsed-literal::

      82	 1.4473346e+00	 1.2672305e-01	 1.4996536e+00	 1.2582645e-01	[ 1.3769032e+00]	 2.1461892e-01


.. parsed-literal::

      83	 1.4501168e+00	 1.2661582e-01	 1.5024616e+00	 1.2569184e-01	[ 1.3784934e+00]	 2.0892358e-01


.. parsed-literal::

      84	 1.4524436e+00	 1.2609625e-01	 1.5050026e+00	 1.2507910e-01	  1.3783082e+00 	 2.1088195e-01


.. parsed-literal::

      85	 1.4573409e+00	 1.2610705e-01	 1.5097579e+00	 1.2513280e-01	[ 1.3826670e+00]	 2.0407844e-01
      86	 1.4587201e+00	 1.2602776e-01	 1.5111065e+00	 1.2512542e-01	[ 1.3836230e+00]	 1.8757558e-01


.. parsed-literal::

      87	 1.4615152e+00	 1.2578745e-01	 1.5139919e+00	 1.2505239e-01	  1.3824185e+00 	 2.1138501e-01


.. parsed-literal::

      88	 1.4628868e+00	 1.2534931e-01	 1.5156504e+00	 1.2487748e-01	  1.3745180e+00 	 2.1607327e-01


.. parsed-literal::

      89	 1.4661773e+00	 1.2528931e-01	 1.5188202e+00	 1.2483873e-01	  1.3784514e+00 	 2.0653677e-01
      90	 1.4677990e+00	 1.2516294e-01	 1.5204595e+00	 1.2474933e-01	  1.3779946e+00 	 1.8244267e-01


.. parsed-literal::

      91	 1.4697686e+00	 1.2496112e-01	 1.5224571e+00	 1.2467508e-01	  1.3773460e+00 	 1.9829869e-01


.. parsed-literal::

      92	 1.4727330e+00	 1.2459363e-01	 1.5254976e+00	 1.2439603e-01	  1.3787211e+00 	 2.0187688e-01
      93	 1.4754626e+00	 1.2427515e-01	 1.5282801e+00	 1.2420967e-01	  1.3804604e+00 	 1.8425322e-01


.. parsed-literal::

      94	 1.4772023e+00	 1.2415379e-01	 1.5299931e+00	 1.2403842e-01	  1.3827475e+00 	 2.1080017e-01


.. parsed-literal::

      95	 1.4797253e+00	 1.2384739e-01	 1.5325594e+00	 1.2367159e-01	[ 1.3866603e+00]	 2.0393610e-01


.. parsed-literal::

      96	 1.4815717e+00	 1.2347132e-01	 1.5344780e+00	 1.2324760e-01	[ 1.3867002e+00]	 2.0628119e-01


.. parsed-literal::

      97	 1.4832666e+00	 1.2335704e-01	 1.5361645e+00	 1.2310525e-01	[ 1.3879911e+00]	 2.0650578e-01


.. parsed-literal::

      98	 1.4851171e+00	 1.2307557e-01	 1.5380531e+00	 1.2298845e-01	  1.3861778e+00 	 2.0712256e-01


.. parsed-literal::

      99	 1.4865379e+00	 1.2296960e-01	 1.5394933e+00	 1.2301726e-01	  1.3849391e+00 	 2.0981526e-01


.. parsed-literal::

     100	 1.4886795e+00	 1.2273420e-01	 1.5416612e+00	 1.2301028e-01	  1.3834769e+00 	 2.1568322e-01


.. parsed-literal::

     101	 1.4904685e+00	 1.2248427e-01	 1.5435383e+00	 1.2304381e-01	  1.3815289e+00 	 2.0686769e-01


.. parsed-literal::

     102	 1.4925046e+00	 1.2238889e-01	 1.5455768e+00	 1.2296735e-01	  1.3833599e+00 	 2.0657444e-01


.. parsed-literal::

     103	 1.4943687e+00	 1.2228194e-01	 1.5474831e+00	 1.2285323e-01	  1.3840566e+00 	 2.0687532e-01


.. parsed-literal::

     104	 1.4964836e+00	 1.2214206e-01	 1.5496685e+00	 1.2274292e-01	  1.3829646e+00 	 2.0381212e-01
     105	 1.4997480e+00	 1.2195352e-01	 1.5531145e+00	 1.2282280e-01	  1.3773255e+00 	 1.7523003e-01


.. parsed-literal::

     106	 1.5015856e+00	 1.2167741e-01	 1.5550641e+00	 1.2264507e-01	  1.3719199e+00 	 2.0834398e-01


.. parsed-literal::

     107	 1.5038315e+00	 1.2165252e-01	 1.5571984e+00	 1.2266077e-01	  1.3735188e+00 	 2.1065521e-01


.. parsed-literal::

     108	 1.5049763e+00	 1.2156627e-01	 1.5583391e+00	 1.2269841e-01	  1.3720806e+00 	 2.0424223e-01


.. parsed-literal::

     109	 1.5064366e+00	 1.2144254e-01	 1.5598252e+00	 1.2277642e-01	  1.3684745e+00 	 2.1285701e-01


.. parsed-literal::

     110	 1.5080879e+00	 1.2131635e-01	 1.5615451e+00	 1.2289884e-01	  1.3628900e+00 	 2.1946049e-01


.. parsed-literal::

     111	 1.5094930e+00	 1.2120737e-01	 1.5629448e+00	 1.2294010e-01	  1.3612817e+00 	 2.0646501e-01


.. parsed-literal::

     112	 1.5107155e+00	 1.2121047e-01	 1.5641764e+00	 1.2302591e-01	  1.3595386e+00 	 2.0666838e-01
     113	 1.5122730e+00	 1.2101079e-01	 1.5657432e+00	 1.2280890e-01	  1.3570628e+00 	 2.0064211e-01


.. parsed-literal::

     114	 1.5142971e+00	 1.2088218e-01	 1.5677763e+00	 1.2278838e-01	  1.3511683e+00 	 2.0843053e-01
     115	 1.5158320e+00	 1.2070635e-01	 1.5692966e+00	 1.2260824e-01	  1.3501158e+00 	 1.8802500e-01


.. parsed-literal::

     116	 1.5177414e+00	 1.2043964e-01	 1.5712258e+00	 1.2243822e-01	  1.3481748e+00 	 1.7868733e-01


.. parsed-literal::

     117	 1.5195152e+00	 1.2016729e-01	 1.5729960e+00	 1.2209920e-01	  1.3470576e+00 	 2.2150779e-01
     118	 1.5213160e+00	 1.1998953e-01	 1.5748825e+00	 1.2195480e-01	  1.3441115e+00 	 1.8787980e-01


.. parsed-literal::

     119	 1.5230878e+00	 1.1975513e-01	 1.5766281e+00	 1.2166932e-01	  1.3450650e+00 	 2.0771933e-01


.. parsed-literal::

     120	 1.5244473e+00	 1.1970495e-01	 1.5779826e+00	 1.2158115e-01	  1.3430000e+00 	 2.1670079e-01
     121	 1.5259980e+00	 1.1965162e-01	 1.5795942e+00	 1.2153115e-01	  1.3394402e+00 	 1.8692875e-01


.. parsed-literal::

     122	 1.5271131e+00	 1.1967874e-01	 1.5808542e+00	 1.2154150e-01	  1.3328265e+00 	 2.0323563e-01
     123	 1.5286508e+00	 1.1958802e-01	 1.5823721e+00	 1.2151971e-01	  1.3329702e+00 	 1.9797611e-01


.. parsed-literal::

     124	 1.5297120e+00	 1.1951849e-01	 1.5834471e+00	 1.2147974e-01	  1.3337285e+00 	 1.8637180e-01


.. parsed-literal::

     125	 1.5308821e+00	 1.1944357e-01	 1.5846937e+00	 1.2144999e-01	  1.3338116e+00 	 2.0801711e-01


.. parsed-literal::

     126	 1.5323153e+00	 1.1939037e-01	 1.5862343e+00	 1.2133813e-01	  1.3316797e+00 	 2.1237874e-01


.. parsed-literal::

     127	 1.5334135e+00	 1.1939507e-01	 1.5873727e+00	 1.2132256e-01	  1.3309872e+00 	 2.1369934e-01
     128	 1.5344774e+00	 1.1943156e-01	 1.5885075e+00	 1.2130652e-01	  1.3254756e+00 	 1.7695236e-01


.. parsed-literal::

     129	 1.5355970e+00	 1.1946180e-01	 1.5896764e+00	 1.2139316e-01	  1.3217986e+00 	 2.1636105e-01


.. parsed-literal::

     130	 1.5369966e+00	 1.1943204e-01	 1.5911218e+00	 1.2145288e-01	  1.3171878e+00 	 2.1657062e-01


.. parsed-literal::

     131	 1.5381125e+00	 1.1949183e-01	 1.5924482e+00	 1.2190596e-01	  1.3035706e+00 	 2.0335436e-01
     132	 1.5396892e+00	 1.1930048e-01	 1.5939348e+00	 1.2173850e-01	  1.3056416e+00 	 1.9565487e-01


.. parsed-literal::

     133	 1.5404749e+00	 1.1920531e-01	 1.5947265e+00	 1.2170938e-01	  1.3048238e+00 	 1.8004322e-01


.. parsed-literal::

     134	 1.5417267e+00	 1.1905033e-01	 1.5960105e+00	 1.2175898e-01	  1.3007500e+00 	 2.1174812e-01


.. parsed-literal::

     135	 1.5436244e+00	 1.1885495e-01	 1.5979277e+00	 1.2188646e-01	  1.2972111e+00 	 2.1006012e-01


.. parsed-literal::

     136	 1.5449775e+00	 1.1861741e-01	 1.5992933e+00	 1.2194957e-01	  1.2898543e+00 	 2.9125977e-01
     137	 1.5463069e+00	 1.1851463e-01	 1.6005990e+00	 1.2206961e-01	  1.2877286e+00 	 1.8165231e-01


.. parsed-literal::

     138	 1.5472614e+00	 1.1842932e-01	 1.6015183e+00	 1.2203785e-01	  1.2886087e+00 	 2.0279455e-01


.. parsed-literal::

     139	 1.5483233e+00	 1.1832812e-01	 1.6025731e+00	 1.2208229e-01	  1.2874065e+00 	 2.0504069e-01
     140	 1.5494058e+00	 1.1818263e-01	 1.6036528e+00	 1.2204016e-01	  1.2836814e+00 	 1.9547772e-01


.. parsed-literal::

     141	 1.5508832e+00	 1.1793839e-01	 1.6051829e+00	 1.2208103e-01	  1.2754054e+00 	 2.1106315e-01
     142	 1.5516472e+00	 1.1777788e-01	 1.6060111e+00	 1.2209615e-01	  1.2622416e+00 	 1.9817281e-01


.. parsed-literal::

     143	 1.5525164e+00	 1.1777812e-01	 1.6068517e+00	 1.2211975e-01	  1.2638027e+00 	 2.0173383e-01


.. parsed-literal::

     144	 1.5537670e+00	 1.1773361e-01	 1.6081239e+00	 1.2224501e-01	  1.2588025e+00 	 2.0611548e-01
     145	 1.5546140e+00	 1.1771420e-01	 1.6089947e+00	 1.2233583e-01	  1.2518949e+00 	 1.7916560e-01


.. parsed-literal::

     146	 1.5559170e+00	 1.1770587e-01	 1.6104142e+00	 1.2270142e-01	  1.2224501e+00 	 2.1381998e-01
     147	 1.5572324e+00	 1.1766636e-01	 1.6117396e+00	 1.2280599e-01	  1.2099296e+00 	 1.8416333e-01


.. parsed-literal::

     148	 1.5578464e+00	 1.1766312e-01	 1.6123013e+00	 1.2274781e-01	  1.2146534e+00 	 2.1757889e-01
     149	 1.5587961e+00	 1.1764085e-01	 1.6132447e+00	 1.2284596e-01	  1.2123310e+00 	 1.9627166e-01


.. parsed-literal::

     150	 1.5596898e+00	 1.1764479e-01	 1.6141765e+00	 1.2290045e-01	  1.2081013e+00 	 2.0755982e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 4s, sys: 1.18 s, total: 2min 5s
    Wall time: 31.5 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fd7e49737f0>



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


.. parsed-literal::

    Process 0 running estimator on chunk 0 - 10,000
    Process 0 estimating GPz PZ PDF for rows 0 - 10,000
    Inserting handle into data store.  output_gpz_run: inprogress_output_gpz_run.hdf5, gpz_run


.. parsed-literal::

    Process 0 running estimator on chunk 10,000 - 20,000
    Process 0 estimating GPz PZ PDF for rows 10,000 - 20,000


.. parsed-literal::

    Process 0 running estimator on chunk 20,000 - 20,449
    Process 0 estimating GPz PZ PDF for rows 20,000 - 20,449
    CPU times: user 2.04 s, sys: 46.9 ms, total: 2.09 s
    Wall time: 618 ms


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

