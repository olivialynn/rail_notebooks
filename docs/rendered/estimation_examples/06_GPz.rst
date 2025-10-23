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
       1	-3.3852593e-01	 3.1925658e-01	-3.2875501e-01	 3.2547526e-01	[-3.4115558e-01]	 4.7070146e-01


.. parsed-literal::

       2	-2.6666358e-01	 3.0779691e-01	-2.4178164e-01	 3.1472881e-01	[-2.6390104e-01]	 2.3404217e-01


.. parsed-literal::

       3	-2.2377821e-01	 2.8807068e-01	-1.8143290e-01	 2.9569868e-01	[-2.1553049e-01]	 2.8643394e-01
       4	-1.9128116e-01	 2.6435490e-01	-1.5067260e-01	 2.7023635e-01	[-1.8525363e-01]	 1.9232941e-01


.. parsed-literal::

       5	-1.0137885e-01	 2.5621828e-01	-6.5619273e-02	 2.5912460e-01	[-8.0829841e-02]	 2.1197462e-01


.. parsed-literal::

       6	-6.7398748e-02	 2.5083576e-01	-3.6552119e-02	 2.5416247e-01	[-4.9281644e-02]	 2.0928550e-01


.. parsed-literal::

       7	-4.8476969e-02	 2.4798905e-01	-2.4776356e-02	 2.5095793e-01	[-3.7180476e-02]	 2.2001004e-01


.. parsed-literal::

       8	-3.6829755e-02	 2.4603716e-01	-1.6607939e-02	 2.4919255e-01	[-3.0174685e-02]	 2.0876718e-01


.. parsed-literal::

       9	-2.3052118e-02	 2.4349106e-01	-5.6863755e-03	 2.4723499e-01	[-2.2321765e-02]	 2.1206880e-01
      10	-1.1872467e-02	 2.4133905e-01	 3.6710882e-03	 2.4596758e-01	[-1.6460743e-02]	 1.8111849e-01


.. parsed-literal::

      11	-8.9631323e-03	 2.4085151e-01	 5.3946562e-03	 2.4583064e-01	 -1.8275223e-02 	 2.1574450e-01


.. parsed-literal::

      12	-4.9533632e-03	 2.4016649e-01	 9.3033510e-03	 2.4501807e-01	[-1.2218257e-02]	 2.0807266e-01


.. parsed-literal::

      13	-2.3893079e-03	 2.3964187e-01	 1.1780752e-02	 2.4440046e-01	[-9.1816220e-03]	 2.1138573e-01


.. parsed-literal::

      14	 1.4432535e-03	 2.3887512e-01	 1.5730098e-02	 2.4354016e-01	[-4.5632562e-03]	 2.1369362e-01


.. parsed-literal::

      15	 1.1455503e-01	 2.2292531e-01	 1.3579341e-01	 2.3073648e-01	[ 1.0055608e-01]	 3.3699608e-01


.. parsed-literal::

      16	 1.5720667e-01	 2.2151630e-01	 1.8127879e-01	 2.3193662e-01	[ 1.4655212e-01]	 2.1149468e-01


.. parsed-literal::

      17	 2.6467609e-01	 2.1617182e-01	 2.9378873e-01	 2.2674940e-01	[ 2.4371236e-01]	 2.1553826e-01


.. parsed-literal::

      18	 3.2238786e-01	 2.1326204e-01	 3.5605113e-01	 2.2126807e-01	[ 2.9215799e-01]	 2.0932150e-01


.. parsed-literal::

      19	 3.5549912e-01	 2.1320942e-01	 3.8957091e-01	 2.2104753e-01	[ 3.2456581e-01]	 2.1183729e-01


.. parsed-literal::

      20	 3.9163168e-01	 2.0937890e-01	 4.2513687e-01	 2.1810483e-01	[ 3.6656942e-01]	 2.2037005e-01


.. parsed-literal::

      21	 4.4433128e-01	 2.0383067e-01	 4.7764405e-01	 2.1561140e-01	[ 4.1419775e-01]	 2.0865297e-01


.. parsed-literal::

      22	 5.2107272e-01	 2.0084254e-01	 5.5497338e-01	 2.1319891e-01	[ 4.9064617e-01]	 2.1456909e-01


.. parsed-literal::

      23	 6.2459025e-01	 1.9724853e-01	 6.6216834e-01	 2.0832750e-01	[ 5.9465068e-01]	 2.1227670e-01


.. parsed-literal::

      24	 6.5504215e-01	 1.9880597e-01	 6.9427663e-01	 2.0906720e-01	[ 6.1563804e-01]	 2.1519828e-01


.. parsed-literal::

      25	 6.8879978e-01	 1.9681010e-01	 7.2701844e-01	 2.0709977e-01	[ 6.6069242e-01]	 2.1779275e-01
      26	 7.0936130e-01	 1.9614782e-01	 7.4763289e-01	 2.0665554e-01	[ 6.8211268e-01]	 1.8223953e-01


.. parsed-literal::

      27	 7.2496122e-01	 2.0441126e-01	 7.6205545e-01	 2.1372121e-01	[ 7.0380144e-01]	 2.0324349e-01
      28	 7.8184428e-01	 2.0021878e-01	 8.2073340e-01	 2.1044813e-01	[ 7.5835654e-01]	 1.9834352e-01


.. parsed-literal::

      29	 8.1747669e-01	 1.9973828e-01	 8.5842625e-01	 2.1150146e-01	[ 7.8568074e-01]	 2.0333171e-01


.. parsed-literal::

      30	 8.4929238e-01	 2.0558476e-01	 8.9100889e-01	 2.1773344e-01	[ 8.1625557e-01]	 2.0673871e-01


.. parsed-literal::

      31	 8.7594256e-01	 2.0770985e-01	 9.1774185e-01	 2.2182520e-01	[ 8.4986400e-01]	 2.1274996e-01


.. parsed-literal::

      32	 8.9645357e-01	 2.0551071e-01	 9.3904321e-01	 2.2101231e-01	[ 8.6892152e-01]	 2.0920563e-01
      33	 9.3011567e-01	 2.0068189e-01	 9.7382115e-01	 2.1660302e-01	[ 9.0093911e-01]	 1.9876528e-01


.. parsed-literal::

      34	 9.3605375e-01	 1.9541708e-01	 9.8328504e-01	 2.1024738e-01	[ 9.1051884e-01]	 2.0270777e-01


.. parsed-literal::

      35	 9.7023570e-01	 1.9282603e-01	 1.0155595e+00	 2.0801718e-01	[ 9.5105606e-01]	 2.1199656e-01
      36	 9.8182504e-01	 1.9186280e-01	 1.0271217e+00	 2.0783202e-01	[ 9.6066988e-01]	 1.9919229e-01


.. parsed-literal::

      37	 1.0038422e+00	 1.8895148e-01	 1.0496112e+00	 2.0545288e-01	[ 9.7941813e-01]	 2.0807528e-01
      38	 1.0219152e+00	 1.8660486e-01	 1.0681102e+00	 2.0422236e-01	[ 9.9538447e-01]	 2.0017862e-01


.. parsed-literal::

      39	 1.0358280e+00	 1.8274570e-01	 1.0825460e+00	 1.9883591e-01	[ 1.0058893e+00]	 2.1165538e-01


.. parsed-literal::

      40	 1.0457810e+00	 1.8102336e-01	 1.0925758e+00	 1.9646695e-01	[ 1.0137601e+00]	 2.0203733e-01


.. parsed-literal::

      41	 1.0607497e+00	 1.7887341e-01	 1.1076169e+00	 1.9372176e-01	[ 1.0294907e+00]	 2.1207094e-01
      42	 1.0728305e+00	 1.7475943e-01	 1.1200455e+00	 1.8956766e-01	[ 1.0423873e+00]	 1.9870734e-01


.. parsed-literal::

      43	 1.0867224e+00	 1.7360254e-01	 1.1337746e+00	 1.8783707e-01	[ 1.0576420e+00]	 1.9680691e-01
      44	 1.0941693e+00	 1.7271715e-01	 1.1413377e+00	 1.8748924e-01	[ 1.0655869e+00]	 1.9254351e-01


.. parsed-literal::

      45	 1.1154771e+00	 1.6908760e-01	 1.1632722e+00	 1.8495485e-01	[ 1.0865466e+00]	 2.1187377e-01


.. parsed-literal::

      46	 1.1361830e+00	 1.6332726e-01	 1.1842847e+00	 1.7848960e-01	[ 1.1087561e+00]	 2.0431972e-01
      47	 1.1542294e+00	 1.5699190e-01	 1.2047724e+00	 1.7014850e-01	[ 1.1217691e+00]	 2.0022774e-01


.. parsed-literal::

      48	 1.1745973e+00	 1.5541293e-01	 1.2239325e+00	 1.6916624e-01	[ 1.1519595e+00]	 2.0258212e-01


.. parsed-literal::

      49	 1.1852538e+00	 1.5342110e-01	 1.2345666e+00	 1.6724826e-01	[ 1.1595879e+00]	 2.0490909e-01


.. parsed-literal::

      50	 1.2032701e+00	 1.5071247e-01	 1.2529076e+00	 1.6411343e-01	[ 1.1703698e+00]	 2.1295691e-01


.. parsed-literal::

      51	 1.2117744e+00	 1.4812170e-01	 1.2611162e+00	 1.6133059e-01	[ 1.1710965e+00]	 2.0224452e-01


.. parsed-literal::

      52	 1.2277281e+00	 1.4753432e-01	 1.2772739e+00	 1.6170278e-01	[ 1.1836446e+00]	 2.1199965e-01


.. parsed-literal::

      53	 1.2392830e+00	 1.4697214e-01	 1.2891934e+00	 1.6120267e-01	[ 1.1913251e+00]	 2.1111131e-01
      54	 1.2500481e+00	 1.4462705e-01	 1.3002497e+00	 1.5994519e-01	[ 1.1943708e+00]	 1.9911814e-01


.. parsed-literal::

      55	 1.2647913e+00	 1.4260255e-01	 1.3149263e+00	 1.5793714e-01	[ 1.1980735e+00]	 2.0777774e-01


.. parsed-literal::

      56	 1.2735177e+00	 1.4028601e-01	 1.3239792e+00	 1.5793978e-01	[ 1.2053675e+00]	 2.1450210e-01


.. parsed-literal::

      57	 1.2858866e+00	 1.3976909e-01	 1.3359498e+00	 1.5731201e-01	[ 1.2178531e+00]	 2.1672297e-01


.. parsed-literal::

      58	 1.2926540e+00	 1.3929650e-01	 1.3428381e+00	 1.5671650e-01	[ 1.2227064e+00]	 2.1898365e-01


.. parsed-literal::

      59	 1.3039526e+00	 1.3849287e-01	 1.3543500e+00	 1.5678249e-01	[ 1.2337969e+00]	 2.1203566e-01


.. parsed-literal::

      60	 1.3158819e+00	 1.3746005e-01	 1.3667361e+00	 1.5673953e-01	[ 1.2426401e+00]	 2.1011925e-01


.. parsed-literal::

      61	 1.3248971e+00	 1.3653309e-01	 1.3765413e+00	 1.5819378e-01	[ 1.2558985e+00]	 2.0898604e-01
      62	 1.3353411e+00	 1.3592524e-01	 1.3867559e+00	 1.5733570e-01	[ 1.2677121e+00]	 1.9816041e-01


.. parsed-literal::

      63	 1.3426657e+00	 1.3569630e-01	 1.3943140e+00	 1.5736457e-01	[ 1.2697495e+00]	 1.9212770e-01
      64	 1.3505687e+00	 1.3477857e-01	 1.4021536e+00	 1.5602961e-01	[ 1.2782286e+00]	 1.8684721e-01


.. parsed-literal::

      65	 1.3594572e+00	 1.3401898e-01	 1.4112570e+00	 1.5602048e-01	[ 1.2845200e+00]	 2.1010113e-01
      66	 1.3664320e+00	 1.3383044e-01	 1.4184086e+00	 1.5590586e-01	[ 1.2885585e+00]	 1.8922710e-01


.. parsed-literal::

      67	 1.3721471e+00	 1.3363626e-01	 1.4240852e+00	 1.5609046e-01	[ 1.2920211e+00]	 2.0955825e-01


.. parsed-literal::

      68	 1.3797520e+00	 1.3328082e-01	 1.4318664e+00	 1.5618033e-01	[ 1.2936382e+00]	 2.0718861e-01
      69	 1.3861926e+00	 1.3262345e-01	 1.4383099e+00	 1.5644306e-01	[ 1.2939767e+00]	 1.8428206e-01


.. parsed-literal::

      70	 1.3906540e+00	 1.3212793e-01	 1.4426191e+00	 1.5546757e-01	[ 1.2983953e+00]	 2.0939088e-01
      71	 1.3980910e+00	 1.3108639e-01	 1.4503624e+00	 1.5438535e-01	  1.2970650e+00 	 1.8644977e-01


.. parsed-literal::

      72	 1.4037287e+00	 1.3055828e-01	 1.4560223e+00	 1.5451630e-01	[ 1.3037986e+00]	 1.9805336e-01


.. parsed-literal::

      73	 1.4088390e+00	 1.3010956e-01	 1.4611838e+00	 1.5383126e-01	[ 1.3046328e+00]	 2.2090912e-01


.. parsed-literal::

      74	 1.4137608e+00	 1.2977404e-01	 1.4661743e+00	 1.5407288e-01	[ 1.3069458e+00]	 2.1823597e-01
      75	 1.4192082e+00	 1.2928882e-01	 1.4717721e+00	 1.5377724e-01	  1.3065083e+00 	 1.7440462e-01


.. parsed-literal::

      76	 1.4251068e+00	 1.2888618e-01	 1.4775991e+00	 1.5435164e-01	[ 1.3111610e+00]	 2.1598744e-01


.. parsed-literal::

      77	 1.4302682e+00	 1.2838368e-01	 1.4828450e+00	 1.5314223e-01	[ 1.3140542e+00]	 2.1377563e-01
      78	 1.4344251e+00	 1.2818713e-01	 1.4870752e+00	 1.5250836e-01	[ 1.3146037e+00]	 2.0714140e-01


.. parsed-literal::

      79	 1.4387349e+00	 1.2792681e-01	 1.4914778e+00	 1.5185777e-01	  1.3136188e+00 	 1.8538260e-01


.. parsed-literal::

      80	 1.4432095e+00	 1.2747306e-01	 1.4958581e+00	 1.5082704e-01	  1.3145796e+00 	 2.2316432e-01
      81	 1.4477403e+00	 1.2714124e-01	 1.5003514e+00	 1.5057580e-01	  1.3108877e+00 	 1.9992042e-01


.. parsed-literal::

      82	 1.4514653e+00	 1.2672707e-01	 1.5040706e+00	 1.5030891e-01	  1.3082815e+00 	 2.0796585e-01


.. parsed-literal::

      83	 1.4539591e+00	 1.2653772e-01	 1.5065516e+00	 1.5018194e-01	  1.3073678e+00 	 2.1161509e-01


.. parsed-literal::

      84	 1.4581535e+00	 1.2617453e-01	 1.5107415e+00	 1.4999364e-01	  1.3071556e+00 	 2.0950913e-01


.. parsed-literal::

      85	 1.4618916e+00	 1.2584409e-01	 1.5147104e+00	 1.4909566e-01	  1.3036842e+00 	 2.1793461e-01


.. parsed-literal::

      86	 1.4658005e+00	 1.2554514e-01	 1.5185766e+00	 1.4909058e-01	  1.3085392e+00 	 2.0708323e-01


.. parsed-literal::

      87	 1.4680181e+00	 1.2545807e-01	 1.5207935e+00	 1.4894553e-01	  1.3112095e+00 	 2.0977521e-01


.. parsed-literal::

      88	 1.4718058e+00	 1.2521445e-01	 1.5246802e+00	 1.4840961e-01	  1.3106973e+00 	 2.2357678e-01


.. parsed-literal::

      89	 1.4739512e+00	 1.2501572e-01	 1.5269332e+00	 1.4782064e-01	  1.3079738e+00 	 3.3070517e-01


.. parsed-literal::

      90	 1.4765639e+00	 1.2484793e-01	 1.5295635e+00	 1.4754056e-01	  1.3064704e+00 	 2.1605396e-01


.. parsed-literal::

      91	 1.4792825e+00	 1.2479898e-01	 1.5323149e+00	 1.4727064e-01	  1.3030351e+00 	 2.0955229e-01
      92	 1.4813860e+00	 1.2504263e-01	 1.5344781e+00	 1.4762747e-01	  1.3003167e+00 	 1.9065356e-01


.. parsed-literal::

      93	 1.4838645e+00	 1.2512447e-01	 1.5369493e+00	 1.4760320e-01	  1.3003711e+00 	 2.0492887e-01


.. parsed-literal::

      94	 1.4866918e+00	 1.2540621e-01	 1.5398245e+00	 1.4771090e-01	  1.2976436e+00 	 2.1149731e-01


.. parsed-literal::

      95	 1.4885760e+00	 1.2551863e-01	 1.5417151e+00	 1.4765597e-01	  1.2959837e+00 	 2.1304011e-01


.. parsed-literal::

      96	 1.4912148e+00	 1.2595413e-01	 1.5444873e+00	 1.4780533e-01	  1.2826462e+00 	 2.1221471e-01
      97	 1.4938549e+00	 1.2580257e-01	 1.5471019e+00	 1.4745431e-01	  1.2831462e+00 	 1.9802141e-01


.. parsed-literal::

      98	 1.4953240e+00	 1.2565909e-01	 1.5485425e+00	 1.4731784e-01	  1.2842723e+00 	 1.8907475e-01


.. parsed-literal::

      99	 1.4980320e+00	 1.2556166e-01	 1.5513119e+00	 1.4703826e-01	  1.2796466e+00 	 2.0728779e-01
     100	 1.5002694e+00	 1.2598676e-01	 1.5537335e+00	 1.4743210e-01	  1.2720637e+00 	 1.9877505e-01


.. parsed-literal::

     101	 1.5025872e+00	 1.2600380e-01	 1.5560080e+00	 1.4718420e-01	  1.2692526e+00 	 2.0756149e-01


.. parsed-literal::

     102	 1.5040555e+00	 1.2617035e-01	 1.5574962e+00	 1.4728419e-01	  1.2666670e+00 	 2.0572090e-01


.. parsed-literal::

     103	 1.5055307e+00	 1.2625845e-01	 1.5590225e+00	 1.4740873e-01	  1.2622453e+00 	 2.1093130e-01
     104	 1.5077849e+00	 1.2632046e-01	 1.5613997e+00	 1.4763299e-01	  1.2536493e+00 	 1.9294095e-01


.. parsed-literal::

     105	 1.5095943e+00	 1.2632081e-01	 1.5632639e+00	 1.4783871e-01	  1.2451438e+00 	 1.7998314e-01


.. parsed-literal::

     106	 1.5108025e+00	 1.2623418e-01	 1.5644189e+00	 1.4779903e-01	  1.2475875e+00 	 2.0477843e-01


.. parsed-literal::

     107	 1.5126583e+00	 1.2614994e-01	 1.5662558e+00	 1.4780217e-01	  1.2482890e+00 	 2.1058440e-01


.. parsed-literal::

     108	 1.5139377e+00	 1.2611431e-01	 1.5675692e+00	 1.4790739e-01	  1.2469327e+00 	 2.2554564e-01


.. parsed-literal::

     109	 1.5154785e+00	 1.2609061e-01	 1.5691089e+00	 1.4791614e-01	  1.2464626e+00 	 2.1076560e-01
     110	 1.5168231e+00	 1.2613873e-01	 1.5704879e+00	 1.4791989e-01	  1.2439173e+00 	 1.9358563e-01


.. parsed-literal::

     111	 1.5181738e+00	 1.2611049e-01	 1.5718761e+00	 1.4781545e-01	  1.2416019e+00 	 2.1693635e-01


.. parsed-literal::

     112	 1.5203983e+00	 1.2593910e-01	 1.5742026e+00	 1.4755833e-01	  1.2360374e+00 	 2.1497750e-01


.. parsed-literal::

     113	 1.5217551e+00	 1.2571670e-01	 1.5755954e+00	 1.4721523e-01	  1.2336927e+00 	 3.3563757e-01
     114	 1.5232743e+00	 1.2553996e-01	 1.5771137e+00	 1.4708358e-01	  1.2321586e+00 	 1.9206142e-01


.. parsed-literal::

     115	 1.5244784e+00	 1.2534028e-01	 1.5783125e+00	 1.4692930e-01	  1.2316564e+00 	 2.0919490e-01


.. parsed-literal::

     116	 1.5256606e+00	 1.2521731e-01	 1.5794924e+00	 1.4691540e-01	  1.2330936e+00 	 2.0192170e-01
     117	 1.5275605e+00	 1.2506078e-01	 1.5814518e+00	 1.4682455e-01	  1.2342087e+00 	 2.0680976e-01


.. parsed-literal::

     118	 1.5292519e+00	 1.2513642e-01	 1.5831511e+00	 1.4703185e-01	  1.2418330e+00 	 1.8379760e-01


.. parsed-literal::

     119	 1.5304836e+00	 1.2523357e-01	 1.5843841e+00	 1.4710480e-01	  1.2444441e+00 	 2.1133780e-01


.. parsed-literal::

     120	 1.5319801e+00	 1.2535244e-01	 1.5859599e+00	 1.4717018e-01	  1.2476729e+00 	 2.0942760e-01


.. parsed-literal::

     121	 1.5330000e+00	 1.2542621e-01	 1.5870620e+00	 1.4725241e-01	  1.2480255e+00 	 2.1203685e-01


.. parsed-literal::

     122	 1.5342042e+00	 1.2533447e-01	 1.5883045e+00	 1.4720822e-01	  1.2471652e+00 	 2.1853161e-01


.. parsed-literal::

     123	 1.5359754e+00	 1.2504451e-01	 1.5901705e+00	 1.4704543e-01	  1.2450039e+00 	 2.2486448e-01


.. parsed-literal::

     124	 1.5370234e+00	 1.2492195e-01	 1.5912437e+00	 1.4713270e-01	  1.2434795e+00 	 2.2364902e-01


.. parsed-literal::

     125	 1.5385755e+00	 1.2476876e-01	 1.5928218e+00	 1.4720637e-01	  1.2444373e+00 	 2.1774244e-01


.. parsed-literal::

     126	 1.5401122e+00	 1.2473347e-01	 1.5944306e+00	 1.4759372e-01	  1.2421288e+00 	 2.1132636e-01


.. parsed-literal::

     127	 1.5414640e+00	 1.2472477e-01	 1.5958058e+00	 1.4754408e-01	  1.2421768e+00 	 2.0776963e-01


.. parsed-literal::

     128	 1.5422020e+00	 1.2477393e-01	 1.5965257e+00	 1.4748790e-01	  1.2422146e+00 	 2.0186353e-01


.. parsed-literal::

     129	 1.5438485e+00	 1.2475797e-01	 1.5982081e+00	 1.4730020e-01	  1.2388413e+00 	 2.1994162e-01


.. parsed-literal::

     130	 1.5442795e+00	 1.2476996e-01	 1.5988218e+00	 1.4715878e-01	  1.2342189e+00 	 2.0617056e-01


.. parsed-literal::

     131	 1.5458162e+00	 1.2465223e-01	 1.6002708e+00	 1.4710562e-01	  1.2349989e+00 	 2.0429921e-01


.. parsed-literal::

     132	 1.5464410e+00	 1.2457051e-01	 1.6009033e+00	 1.4706821e-01	  1.2350023e+00 	 2.0998240e-01


.. parsed-literal::

     133	 1.5474834e+00	 1.2446079e-01	 1.6019774e+00	 1.4700481e-01	  1.2353413e+00 	 2.0291328e-01


.. parsed-literal::

     134	 1.5488292e+00	 1.2440650e-01	 1.6033463e+00	 1.4699146e-01	  1.2354318e+00 	 2.0423126e-01


.. parsed-literal::

     135	 1.5498131e+00	 1.2428088e-01	 1.6043862e+00	 1.4694856e-01	  1.2337185e+00 	 2.1048117e-01


.. parsed-literal::

     136	 1.5510726e+00	 1.2434898e-01	 1.6055684e+00	 1.4696456e-01	  1.2352035e+00 	 2.0870304e-01


.. parsed-literal::

     137	 1.5519958e+00	 1.2442392e-01	 1.6064580e+00	 1.4699245e-01	  1.2354896e+00 	 2.1176648e-01


.. parsed-literal::

     138	 1.5531299e+00	 1.2446368e-01	 1.6075875e+00	 1.4706910e-01	  1.2358692e+00 	 2.1816659e-01


.. parsed-literal::

     139	 1.5540494e+00	 1.2454594e-01	 1.6085887e+00	 1.4705860e-01	  1.2368649e+00 	 2.1084094e-01


.. parsed-literal::

     140	 1.5553111e+00	 1.2443876e-01	 1.6098101e+00	 1.4700759e-01	  1.2380110e+00 	 2.0702648e-01


.. parsed-literal::

     141	 1.5559916e+00	 1.2433233e-01	 1.6104944e+00	 1.4692316e-01	  1.2383771e+00 	 2.1174455e-01


.. parsed-literal::

     142	 1.5569492e+00	 1.2420236e-01	 1.6114777e+00	 1.4678296e-01	  1.2372644e+00 	 2.1336365e-01


.. parsed-literal::

     143	 1.5587865e+00	 1.2396532e-01	 1.6133921e+00	 1.4655348e-01	  1.2338985e+00 	 2.1007371e-01


.. parsed-literal::

     144	 1.5597069e+00	 1.2388972e-01	 1.6143663e+00	 1.4642655e-01	  1.2258098e+00 	 3.1865382e-01


.. parsed-literal::

     145	 1.5610192e+00	 1.2379751e-01	 1.6157162e+00	 1.4634283e-01	  1.2229261e+00 	 2.1354389e-01
     146	 1.5620749e+00	 1.2376484e-01	 1.6167946e+00	 1.4633966e-01	  1.2210886e+00 	 2.0224190e-01


.. parsed-literal::

     147	 1.5632043e+00	 1.2371202e-01	 1.6179423e+00	 1.4640209e-01	  1.2189801e+00 	 2.1993756e-01


.. parsed-literal::

     148	 1.5639089e+00	 1.2362651e-01	 1.6187546e+00	 1.4639876e-01	  1.2200165e+00 	 2.1811485e-01
     149	 1.5655438e+00	 1.2354928e-01	 1.6203010e+00	 1.4647657e-01	  1.2182687e+00 	 1.9016862e-01


.. parsed-literal::

     150	 1.5662779e+00	 1.2348904e-01	 1.6210216e+00	 1.4648619e-01	  1.2173189e+00 	 1.8700862e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 6s, sys: 1.17 s, total: 2min 8s
    Wall time: 32.1 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f2e407df9d0>



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
    CPU times: user 1.88 s, sys: 51 ms, total: 1.93 s
    Wall time: 646 ms


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

