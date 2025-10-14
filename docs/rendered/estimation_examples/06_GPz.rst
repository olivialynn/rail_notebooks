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
       1	-3.3305968e-01	 3.1739976e-01	-3.2323402e-01	 3.3423083e-01	[-3.5501334e-01]	 4.6255064e-01


.. parsed-literal::

       2	-2.6101034e-01	 3.0611544e-01	-2.3658727e-01	 3.2149950e-01	[-2.8494525e-01]	 2.2974277e-01


.. parsed-literal::

       3	-2.1688926e-01	 2.8665408e-01	-1.7577006e-01	 3.0283114e-01	[-2.4251158e-01]	 2.7982068e-01


.. parsed-literal::

       4	-1.7655273e-01	 2.6872369e-01	-1.2801299e-01	 2.8444082e-01	[-2.0422677e-01]	 3.1298351e-01


.. parsed-literal::

       5	-1.3143234e-01	 2.5536627e-01	-1.0119534e-01	 2.6813400e-01	[-1.7801338e-01]	 2.1022820e-01


.. parsed-literal::

       6	-6.6164772e-02	 2.5126187e-01	-3.7174138e-02	 2.6430102e-01	[-9.1927743e-02]	 2.1965384e-01
       7	-4.5676674e-02	 2.4701235e-01	-2.1311008e-02	 2.6079320e-01	[-7.6105944e-02]	 1.8842554e-01


.. parsed-literal::

       8	-3.3883275e-02	 2.4494859e-01	-1.2387063e-02	 2.5935709e-01	[-7.2006179e-02]	 1.7737699e-01


.. parsed-literal::

       9	-1.8653522e-02	 2.4224622e-01	-7.7646880e-04	 2.5789046e-01	[-6.7354959e-02]	 2.1463561e-01


.. parsed-literal::

      10	-7.1744406e-03	 2.4008769e-01	 8.7216509e-03	 2.5592842e-01	[-6.0864711e-02]	 2.1363592e-01
      11	 1.3411957e-03	 2.3868538e-01	 1.5545977e-02	 2.5383926e-01	[-5.0695202e-02]	 1.7384911e-01


.. parsed-literal::

      12	 5.6772208e-03	 2.3782908e-01	 1.9617256e-02	 2.5355448e-01	[-4.7341067e-02]	 2.0951009e-01
      13	 9.0982127e-03	 2.3717882e-01	 2.3005227e-02	 2.5259749e-01	[-4.3047145e-02]	 1.8697190e-01


.. parsed-literal::

      14	 1.0800264e-01	 2.2001810e-01	 1.2971621e-01	 2.3453616e-01	[ 1.0320601e-01]	 4.3915820e-01


.. parsed-literal::

      15	 2.1828221e-01	 2.1604980e-01	 2.4186630e-01	 2.3210572e-01	[ 1.9619327e-01]	 2.0689678e-01


.. parsed-literal::

      16	 2.6793018e-01	 2.1095270e-01	 2.9922314e-01	 2.2413866e-01	[ 2.2781161e-01]	 2.1255803e-01
      17	 3.1631060e-01	 2.1039979e-01	 3.4743473e-01	 2.2333065e-01	[ 2.9649670e-01]	 1.8561959e-01


.. parsed-literal::

      18	 3.5285757e-01	 2.0811349e-01	 3.8573763e-01	 2.2396498e-01	[ 3.4254683e-01]	 2.1094012e-01


.. parsed-literal::

      19	 3.9790467e-01	 2.0457469e-01	 4.3119727e-01	 2.2115736e-01	[ 3.8825250e-01]	 2.1319914e-01


.. parsed-literal::

      20	 4.6040540e-01	 2.0327215e-01	 4.9352662e-01	 2.1827711e-01	[ 4.5125561e-01]	 2.1059704e-01
      21	 5.2765999e-01	 1.9996546e-01	 5.6275839e-01	 2.1519283e-01	[ 5.2150997e-01]	 1.7870522e-01


.. parsed-literal::

      22	 5.8307400e-01	 1.9858445e-01	 6.2031106e-01	 2.1466281e-01	[ 5.7952995e-01]	 2.0497227e-01


.. parsed-literal::

      23	 6.4541446e-01	 1.9808825e-01	 6.8511686e-01	 2.1376546e-01	[ 6.4034923e-01]	 2.1004152e-01


.. parsed-literal::

      24	 6.6535697e-01	 1.9434101e-01	 7.0469547e-01	 2.1151675e-01	[ 6.6418585e-01]	 2.0702481e-01


.. parsed-literal::

      25	 6.9412686e-01	 1.9072204e-01	 7.3328446e-01	 2.0776434e-01	[ 6.9896604e-01]	 2.0960021e-01


.. parsed-literal::

      26	 7.2893472e-01	 1.8964801e-01	 7.6813713e-01	 2.0822892e-01	[ 7.3695258e-01]	 2.0925689e-01


.. parsed-literal::

      27	 7.6423701e-01	 1.8697474e-01	 8.0260354e-01	 2.0505074e-01	[ 7.7774221e-01]	 2.1267033e-01


.. parsed-literal::

      28	 7.8973579e-01	 1.8915115e-01	 8.2797383e-01	 2.0823708e-01	[ 8.0343516e-01]	 2.0684600e-01


.. parsed-literal::

      29	 8.1774430e-01	 1.8756965e-01	 8.5641423e-01	 2.0915942e-01	[ 8.2744833e-01]	 2.2079372e-01


.. parsed-literal::

      30	 8.3709206e-01	 1.8504813e-01	 8.7649056e-01	 2.0562965e-01	[ 8.4820880e-01]	 2.0934200e-01


.. parsed-literal::

      31	 8.5791749e-01	 1.8314664e-01	 8.9762765e-01	 2.0322380e-01	[ 8.6644886e-01]	 2.1031332e-01


.. parsed-literal::

      32	 8.8343223e-01	 1.8008664e-01	 9.2445863e-01	 1.9841721e-01	[ 8.9016765e-01]	 2.1761847e-01


.. parsed-literal::

      33	 9.0307926e-01	 1.7926771e-01	 9.4428512e-01	 1.9910436e-01	[ 9.0966382e-01]	 2.0443845e-01


.. parsed-literal::

      34	 9.2148725e-01	 1.7672153e-01	 9.6296999e-01	 1.9792163e-01	[ 9.3126207e-01]	 2.0592380e-01
      35	 9.4012990e-01	 1.7511830e-01	 9.8218737e-01	 1.9758269e-01	[ 9.4833377e-01]	 2.0114779e-01


.. parsed-literal::

      36	 9.6096276e-01	 1.7339583e-01	 1.0039345e+00	 1.9643643e-01	[ 9.6629115e-01]	 2.0125198e-01


.. parsed-literal::

      37	 9.7911767e-01	 1.7259737e-01	 1.0227317e+00	 1.9727033e-01	[ 9.7783404e-01]	 2.0350337e-01


.. parsed-literal::

      38	 9.9982344e-01	 1.7185507e-01	 1.0442216e+00	 1.9721369e-01	[ 9.8944112e-01]	 2.1214414e-01
      39	 1.0190171e+00	 1.7022421e-01	 1.0638464e+00	 1.9579533e-01	[ 1.0083980e+00]	 1.7298889e-01


.. parsed-literal::

      40	 1.0349086e+00	 1.6809477e-01	 1.0799888e+00	 1.9297581e-01	[ 1.0240723e+00]	 2.1229649e-01
      41	 1.0545205e+00	 1.6410693e-01	 1.1007740e+00	 1.8815144e-01	[ 1.0399956e+00]	 1.9669628e-01


.. parsed-literal::

      42	 1.0655785e+00	 1.6265397e-01	 1.1129085e+00	 1.8575009e-01	  1.0367702e+00 	 1.9188333e-01


.. parsed-literal::

      43	 1.0766959e+00	 1.6134847e-01	 1.1236879e+00	 1.8468772e-01	[ 1.0480731e+00]	 2.2002339e-01
      44	 1.0864547e+00	 1.6015356e-01	 1.1335561e+00	 1.8358687e-01	[ 1.0574728e+00]	 2.0002484e-01


.. parsed-literal::

      45	 1.0974800e+00	 1.5917026e-01	 1.1445551e+00	 1.8275865e-01	[ 1.0695153e+00]	 2.0808840e-01


.. parsed-literal::

      46	 1.1102686e+00	 1.5669507e-01	 1.1575341e+00	 1.7893008e-01	[ 1.0821602e+00]	 2.1677184e-01


.. parsed-literal::

      47	 1.1232340e+00	 1.5531235e-01	 1.1703134e+00	 1.7720377e-01	[ 1.0939500e+00]	 2.1122193e-01


.. parsed-literal::

      48	 1.1334106e+00	 1.5353819e-01	 1.1807782e+00	 1.7465858e-01	[ 1.1005837e+00]	 2.1047974e-01


.. parsed-literal::

      49	 1.1475780e+00	 1.5099822e-01	 1.1950080e+00	 1.7020880e-01	[ 1.1171770e+00]	 2.1267676e-01


.. parsed-literal::

      50	 1.1614371e+00	 1.4817664e-01	 1.2090130e+00	 1.6542503e-01	[ 1.1319598e+00]	 2.1227050e-01


.. parsed-literal::

      51	 1.1724835e+00	 1.4631379e-01	 1.2201207e+00	 1.6225949e-01	[ 1.1458743e+00]	 2.1181560e-01


.. parsed-literal::

      52	 1.1876918e+00	 1.4334508e-01	 1.2358560e+00	 1.5769398e-01	[ 1.1606186e+00]	 2.1332335e-01


.. parsed-literal::

      53	 1.1943243e+00	 1.4174295e-01	 1.2427239e+00	 1.5462026e-01	[ 1.1764116e+00]	 2.1610284e-01


.. parsed-literal::

      54	 1.2026811e+00	 1.4098633e-01	 1.2509988e+00	 1.5436927e-01	[ 1.1818150e+00]	 2.1161079e-01


.. parsed-literal::

      55	 1.2100580e+00	 1.4019732e-01	 1.2585154e+00	 1.5402986e-01	[ 1.1865906e+00]	 2.1318817e-01


.. parsed-literal::

      56	 1.2165643e+00	 1.3974143e-01	 1.2651207e+00	 1.5356825e-01	[ 1.1922064e+00]	 2.0965052e-01


.. parsed-literal::

      57	 1.2332759e+00	 1.3879792e-01	 1.2823562e+00	 1.5167655e-01	[ 1.2048692e+00]	 2.0588136e-01


.. parsed-literal::

      58	 1.2360551e+00	 1.3857421e-01	 1.2858512e+00	 1.4995071e-01	  1.2031423e+00 	 2.2418714e-01
      59	 1.2512128e+00	 1.3749806e-01	 1.3004793e+00	 1.4915614e-01	[ 1.2205986e+00]	 1.8834972e-01


.. parsed-literal::

      60	 1.2571605e+00	 1.3672291e-01	 1.3064581e+00	 1.4855030e-01	[ 1.2248946e+00]	 2.1683097e-01


.. parsed-literal::

      61	 1.2654482e+00	 1.3552272e-01	 1.3150254e+00	 1.4781339e-01	[ 1.2275731e+00]	 2.0889592e-01
      62	 1.2726677e+00	 1.3445305e-01	 1.3224698e+00	 1.4740819e-01	  1.2273739e+00 	 1.7562437e-01


.. parsed-literal::

      63	 1.2799665e+00	 1.3399584e-01	 1.3297158e+00	 1.4731852e-01	[ 1.2312852e+00]	 2.0126176e-01


.. parsed-literal::

      64	 1.2869112e+00	 1.3352256e-01	 1.3367690e+00	 1.4700864e-01	[ 1.2340264e+00]	 2.0873308e-01


.. parsed-literal::

      65	 1.2939989e+00	 1.3311716e-01	 1.3439445e+00	 1.4675821e-01	[ 1.2370883e+00]	 2.0314527e-01


.. parsed-literal::

      66	 1.3023749e+00	 1.3217262e-01	 1.3526213e+00	 1.4582749e-01	[ 1.2407503e+00]	 2.1029162e-01


.. parsed-literal::

      67	 1.3105509e+00	 1.3160905e-01	 1.3608401e+00	 1.4579089e-01	[ 1.2493402e+00]	 2.0413613e-01
      68	 1.3176697e+00	 1.3098366e-01	 1.3680596e+00	 1.4552001e-01	[ 1.2565334e+00]	 2.0192051e-01


.. parsed-literal::

      69	 1.3248891e+00	 1.3026638e-01	 1.3756472e+00	 1.4632357e-01	[ 1.2584385e+00]	 2.0682645e-01


.. parsed-literal::

      70	 1.3307537e+00	 1.2984846e-01	 1.3814747e+00	 1.4578309e-01	[ 1.2658876e+00]	 2.0592546e-01


.. parsed-literal::

      71	 1.3341038e+00	 1.2964398e-01	 1.3847084e+00	 1.4568204e-01	[ 1.2689111e+00]	 2.0585871e-01
      72	 1.3429008e+00	 1.2892212e-01	 1.3936720e+00	 1.4614298e-01	  1.2664041e+00 	 1.6482759e-01


.. parsed-literal::

      73	 1.3469949e+00	 1.2862518e-01	 1.3979852e+00	 1.4628338e-01	  1.2683754e+00 	 1.7914653e-01


.. parsed-literal::

      74	 1.3525665e+00	 1.2844203e-01	 1.4034182e+00	 1.4610490e-01	[ 1.2738371e+00]	 2.1833563e-01


.. parsed-literal::

      75	 1.3583806e+00	 1.2824048e-01	 1.4093105e+00	 1.4609695e-01	[ 1.2763638e+00]	 2.1234798e-01


.. parsed-literal::

      76	 1.3626058e+00	 1.2807142e-01	 1.4136122e+00	 1.4584228e-01	[ 1.2779358e+00]	 2.0832181e-01


.. parsed-literal::

      77	 1.3702225e+00	 1.2740359e-01	 1.4217008e+00	 1.4571402e-01	  1.2778905e+00 	 2.1250534e-01
      78	 1.3772417e+00	 1.2692098e-01	 1.4287901e+00	 1.4476163e-01	  1.2766856e+00 	 1.8602824e-01


.. parsed-literal::

      79	 1.3809648e+00	 1.2666446e-01	 1.4323722e+00	 1.4474169e-01	[ 1.2828613e+00]	 2.0710707e-01
      80	 1.3873524e+00	 1.2607633e-01	 1.4388728e+00	 1.4487903e-01	[ 1.2877806e+00]	 1.8808889e-01


.. parsed-literal::

      81	 1.3931395e+00	 1.2549390e-01	 1.4448401e+00	 1.4493264e-01	[ 1.2912770e+00]	 2.0168829e-01


.. parsed-literal::

      82	 1.3988699e+00	 1.2501407e-01	 1.4507454e+00	 1.4485999e-01	[ 1.2963972e+00]	 2.0431876e-01


.. parsed-literal::

      83	 1.4035452e+00	 1.2499554e-01	 1.4554366e+00	 1.4468704e-01	[ 1.3011235e+00]	 2.0741439e-01
      84	 1.4086262e+00	 1.2472287e-01	 1.4606281e+00	 1.4363470e-01	[ 1.3056631e+00]	 1.8475914e-01


.. parsed-literal::

      85	 1.4119801e+00	 1.2465941e-01	 1.4641833e+00	 1.4394196e-01	  1.3049445e+00 	 2.1648359e-01


.. parsed-literal::

      86	 1.4153022e+00	 1.2445242e-01	 1.4674428e+00	 1.4365092e-01	[ 1.3090830e+00]	 2.1136045e-01
      87	 1.4199428e+00	 1.2402607e-01	 1.4722362e+00	 1.4314373e-01	[ 1.3130193e+00]	 1.9562340e-01


.. parsed-literal::

      88	 1.4228190e+00	 1.2386509e-01	 1.4750302e+00	 1.4335263e-01	[ 1.3149659e+00]	 2.0527196e-01


.. parsed-literal::

      89	 1.4259113e+00	 1.2373344e-01	 1.4781137e+00	 1.4351025e-01	[ 1.3164147e+00]	 2.1238184e-01
      90	 1.4319278e+00	 1.2351836e-01	 1.4841867e+00	 1.4393645e-01	[ 1.3200689e+00]	 1.8696666e-01


.. parsed-literal::

      91	 1.4370318e+00	 1.2353456e-01	 1.4895686e+00	 1.4405190e-01	  1.3197078e+00 	 2.0789361e-01


.. parsed-literal::

      92	 1.4423393e+00	 1.2370497e-01	 1.4950166e+00	 1.4519559e-01	  1.3191424e+00 	 2.1887922e-01


.. parsed-literal::

      93	 1.4461150e+00	 1.2359867e-01	 1.4988130e+00	 1.4497140e-01	[ 1.3210314e+00]	 2.1217251e-01


.. parsed-literal::

      94	 1.4492749e+00	 1.2346703e-01	 1.5020126e+00	 1.4476670e-01	[ 1.3212821e+00]	 2.2191525e-01


.. parsed-literal::

      95	 1.4530682e+00	 1.2315739e-01	 1.5058409e+00	 1.4433314e-01	[ 1.3213687e+00]	 2.1165466e-01
      96	 1.4569617e+00	 1.2275324e-01	 1.5099778e+00	 1.4365914e-01	  1.3126599e+00 	 1.8277335e-01


.. parsed-literal::

      97	 1.4616577e+00	 1.2258264e-01	 1.5145857e+00	 1.4368735e-01	  1.3198000e+00 	 1.7462230e-01


.. parsed-literal::

      98	 1.4637819e+00	 1.2247045e-01	 1.5165792e+00	 1.4368498e-01	[ 1.3233037e+00]	 2.1129394e-01


.. parsed-literal::

      99	 1.4670979e+00	 1.2229564e-01	 1.5199113e+00	 1.4426332e-01	  1.3190917e+00 	 2.1305919e-01


.. parsed-literal::

     100	 1.4698947e+00	 1.2206453e-01	 1.5228021e+00	 1.4514442e-01	  1.3133692e+00 	 2.2000241e-01
     101	 1.4727611e+00	 1.2193814e-01	 1.5257034e+00	 1.4532005e-01	  1.3106123e+00 	 2.0799637e-01


.. parsed-literal::

     102	 1.4757791e+00	 1.2180412e-01	 1.5287607e+00	 1.4575021e-01	  1.3068059e+00 	 2.0828700e-01


.. parsed-literal::

     103	 1.4784221e+00	 1.2177622e-01	 1.5314720e+00	 1.4545725e-01	  1.3084817e+00 	 2.1362638e-01


.. parsed-literal::

     104	 1.4809735e+00	 1.2155694e-01	 1.5340203e+00	 1.4544514e-01	  1.3078584e+00 	 2.1302533e-01


.. parsed-literal::

     105	 1.4826130e+00	 1.2149547e-01	 1.5356313e+00	 1.4504052e-01	  1.3077470e+00 	 2.1761584e-01


.. parsed-literal::

     106	 1.4855549e+00	 1.2145371e-01	 1.5386246e+00	 1.4427133e-01	  1.3073037e+00 	 2.0775294e-01


.. parsed-literal::

     107	 1.4874817e+00	 1.2133921e-01	 1.5407626e+00	 1.4359641e-01	  1.2882416e+00 	 2.1959829e-01


.. parsed-literal::

     108	 1.4904253e+00	 1.2132172e-01	 1.5436247e+00	 1.4366412e-01	  1.2973344e+00 	 2.2054148e-01


.. parsed-literal::

     109	 1.4922574e+00	 1.2126697e-01	 1.5454774e+00	 1.4395874e-01	  1.2982449e+00 	 2.1354771e-01


.. parsed-literal::

     110	 1.4939668e+00	 1.2121277e-01	 1.5472149e+00	 1.4395930e-01	  1.3002587e+00 	 2.2021103e-01


.. parsed-literal::

     111	 1.4967149e+00	 1.2112758e-01	 1.5500704e+00	 1.4417653e-01	  1.2927899e+00 	 2.1466613e-01


.. parsed-literal::

     112	 1.4991221e+00	 1.2105863e-01	 1.5525806e+00	 1.4363870e-01	  1.3017194e+00 	 2.0650268e-01


.. parsed-literal::

     113	 1.5009031e+00	 1.2106025e-01	 1.5542976e+00	 1.4359591e-01	  1.3002824e+00 	 2.1589899e-01


.. parsed-literal::

     114	 1.5032314e+00	 1.2103505e-01	 1.5566496e+00	 1.4359208e-01	  1.2938552e+00 	 2.1302557e-01
     115	 1.5052331e+00	 1.2102590e-01	 1.5587462e+00	 1.4383283e-01	  1.2904405e+00 	 1.9972992e-01


.. parsed-literal::

     116	 1.5078860e+00	 1.2089059e-01	 1.5615687e+00	 1.4465465e-01	  1.2829241e+00 	 2.0451760e-01


.. parsed-literal::

     117	 1.5094692e+00	 1.2095732e-01	 1.5631966e+00	 1.4489132e-01	  1.2894266e+00 	 2.1155787e-01


.. parsed-literal::

     118	 1.5108662e+00	 1.2087132e-01	 1.5645096e+00	 1.4494512e-01	  1.2937182e+00 	 2.0300603e-01
     119	 1.5125407e+00	 1.2079250e-01	 1.5661159e+00	 1.4506698e-01	  1.2979139e+00 	 1.7942953e-01


.. parsed-literal::

     120	 1.5145443e+00	 1.2070794e-01	 1.5680857e+00	 1.4514659e-01	  1.2989388e+00 	 1.8859196e-01


.. parsed-literal::

     121	 1.5162432e+00	 1.2073187e-01	 1.5698581e+00	 1.4544243e-01	  1.3061331e+00 	 2.1089578e-01


.. parsed-literal::

     122	 1.5180259e+00	 1.2071516e-01	 1.5716058e+00	 1.4547595e-01	  1.3025863e+00 	 2.0204353e-01


.. parsed-literal::

     123	 1.5196226e+00	 1.2069915e-01	 1.5732487e+00	 1.4570573e-01	  1.2982278e+00 	 2.0351744e-01


.. parsed-literal::

     124	 1.5212281e+00	 1.2066232e-01	 1.5749088e+00	 1.4605910e-01	  1.2956706e+00 	 2.0839882e-01


.. parsed-literal::

     125	 1.5243642e+00	 1.2054536e-01	 1.5781371e+00	 1.4679942e-01	  1.2895211e+00 	 2.0908761e-01


.. parsed-literal::

     126	 1.5256216e+00	 1.2052110e-01	 1.5794287e+00	 1.4697883e-01	  1.2852210e+00 	 3.2758069e-01


.. parsed-literal::

     127	 1.5270876e+00	 1.2040612e-01	 1.5808634e+00	 1.4692157e-01	  1.2847996e+00 	 2.0922685e-01
     128	 1.5283934e+00	 1.2020858e-01	 1.5821254e+00	 1.4643035e-01	  1.2859677e+00 	 1.8779397e-01


.. parsed-literal::

     129	 1.5295656e+00	 1.2004030e-01	 1.5832611e+00	 1.4589744e-01	  1.2741476e+00 	 2.0027232e-01


.. parsed-literal::

     130	 1.5311265e+00	 1.1989086e-01	 1.5847960e+00	 1.4519975e-01	  1.2754673e+00 	 2.1897602e-01


.. parsed-literal::

     131	 1.5325376e+00	 1.1977106e-01	 1.5862259e+00	 1.4459259e-01	  1.2704325e+00 	 2.1091485e-01
     132	 1.5334926e+00	 1.1976588e-01	 1.5872522e+00	 1.4439779e-01	  1.2628360e+00 	 2.0363235e-01


.. parsed-literal::

     133	 1.5347401e+00	 1.1973386e-01	 1.5886082e+00	 1.4430287e-01	  1.2546937e+00 	 1.9844389e-01


.. parsed-literal::

     134	 1.5361726e+00	 1.1973874e-01	 1.5901391e+00	 1.4435112e-01	  1.2493246e+00 	 2.0961952e-01
     135	 1.5378067e+00	 1.1980058e-01	 1.5919397e+00	 1.4450708e-01	  1.2424494e+00 	 1.9223142e-01


.. parsed-literal::

     136	 1.5388014e+00	 1.1977858e-01	 1.5930390e+00	 1.4446927e-01	  1.2479806e+00 	 1.9875741e-01


.. parsed-literal::

     137	 1.5397757e+00	 1.1975919e-01	 1.5939142e+00	 1.4445513e-01	  1.2502612e+00 	 2.0849800e-01


.. parsed-literal::

     138	 1.5406746e+00	 1.1973562e-01	 1.5947726e+00	 1.4439728e-01	  1.2513618e+00 	 2.1464705e-01


.. parsed-literal::

     139	 1.5415802e+00	 1.1971257e-01	 1.5956681e+00	 1.4446303e-01	  1.2528013e+00 	 2.1812820e-01


.. parsed-literal::

     140	 1.5434205e+00	 1.1966164e-01	 1.5975129e+00	 1.4478237e-01	  1.2544275e+00 	 2.0734024e-01


.. parsed-literal::

     141	 1.5440930e+00	 1.1953927e-01	 1.5983746e+00	 1.4513200e-01	  1.2478371e+00 	 2.0879674e-01


.. parsed-literal::

     142	 1.5465309e+00	 1.1947070e-01	 1.6008063e+00	 1.4515849e-01	  1.2497780e+00 	 2.1804571e-01


.. parsed-literal::

     143	 1.5473399e+00	 1.1940744e-01	 1.6016008e+00	 1.4510953e-01	  1.2475979e+00 	 2.0460463e-01
     144	 1.5482365e+00	 1.1927747e-01	 1.6026616e+00	 1.4494554e-01	  1.2400255e+00 	 1.8326044e-01


.. parsed-literal::

     145	 1.5487758e+00	 1.1915898e-01	 1.6033026e+00	 1.4482672e-01	  1.2338959e+00 	 2.0478821e-01


.. parsed-literal::

     146	 1.5495657e+00	 1.1915466e-01	 1.6040626e+00	 1.4469730e-01	  1.2336017e+00 	 2.1095061e-01


.. parsed-literal::

     147	 1.5506199e+00	 1.1910646e-01	 1.6051736e+00	 1.4440108e-01	  1.2303345e+00 	 2.0673394e-01


.. parsed-literal::

     148	 1.5514280e+00	 1.1905211e-01	 1.6060293e+00	 1.4418588e-01	  1.2294418e+00 	 2.2181129e-01
     149	 1.5526108e+00	 1.1900397e-01	 1.6073526e+00	 1.4390174e-01	  1.2209172e+00 	 2.0844984e-01


.. parsed-literal::

     150	 1.5542649e+00	 1.1892756e-01	 1.6089153e+00	 1.4378783e-01	  1.2315468e+00 	 1.8372416e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 6s, sys: 1.08 s, total: 2min 7s
    Wall time: 31.9 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f2d34d46680>



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
    CPU times: user 1.74 s, sys: 56 ms, total: 1.8 s
    Wall time: 566 ms


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

