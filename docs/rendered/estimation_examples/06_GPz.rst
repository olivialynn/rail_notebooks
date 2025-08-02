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
       1	-3.3483414e-01	 3.1759379e-01	-3.2514901e-01	 3.3257892e-01	[-3.5389074e-01]	 4.6176815e-01


.. parsed-literal::

       2	-2.6275825e-01	 3.0665318e-01	-2.3815599e-01	 3.2164156e-01	[-2.8521645e-01]	 2.2977185e-01


.. parsed-literal::

       3	-2.1777828e-01	 2.8604614e-01	-1.7480860e-01	 3.0353932e-01	[-2.4971058e-01]	 2.8614378e-01
       4	-1.8860067e-01	 2.6174591e-01	-1.4587251e-01	 2.8358994e-01	 -2.7734394e-01 	 1.7027497e-01


.. parsed-literal::

       5	-9.5559590e-02	 2.5438638e-01	-5.8882062e-02	 2.7314668e-01	[-1.5119798e-01]	 2.0569324e-01
       6	-5.8664084e-02	 2.4808954e-01	-2.6175222e-02	 2.6895982e-01	[-1.0610915e-01]	 1.7807794e-01


.. parsed-literal::

       7	-4.0525164e-02	 2.4546816e-01	-1.4728359e-02	 2.6578752e-01	[-9.8417295e-02]	 2.0966887e-01
       8	-2.6928434e-02	 2.4327052e-01	-5.4406436e-03	 2.6352611e-01	[-9.1962739e-02]	 1.7890382e-01


.. parsed-literal::

       9	-1.1236065e-02	 2.4036283e-01	 7.0031961e-03	 2.6104983e-01	[-8.1172800e-02]	 1.8995309e-01
      10	 3.4925632e-04	 2.3799322e-01	 1.6079624e-02	 2.5995958e-01	 -8.6387656e-02 	 1.9179368e-01


.. parsed-literal::

      11	 6.6440061e-03	 2.3722461e-01	 2.0843612e-02	 2.5842722e-01	[-6.7355975e-02]	 1.9859409e-01


.. parsed-literal::

      12	 9.6122230e-03	 2.3674005e-01	 2.3533170e-02	 2.5776481e-01	[-6.6227249e-02]	 2.1030951e-01
      13	 1.3933471e-02	 2.3594037e-01	 2.7720837e-02	 2.5636242e-01	[-6.1556267e-02]	 1.6760206e-01


.. parsed-literal::

      14	 1.6517789e-01	 2.2242230e-01	 1.8873319e-01	 2.4200972e-01	[ 1.2403083e-01]	 4.2443299e-01


.. parsed-literal::

      15	 1.8925490e-01	 2.1924318e-01	 2.1219428e-01	 2.4023658e-01	[ 1.5535662e-01]	 3.1605339e-01


.. parsed-literal::

      16	 2.6231281e-01	 2.1606452e-01	 2.9005635e-01	 2.3453340e-01	[ 2.2446960e-01]	 2.0612097e-01
      17	 3.0342155e-01	 2.1326620e-01	 3.3829874e-01	 2.3165375e-01	[ 2.4501671e-01]	 1.9765973e-01


.. parsed-literal::

      18	 3.5971407e-01	 2.0910523e-01	 3.9365415e-01	 2.2337008e-01	[ 3.0153288e-01]	 2.1203041e-01


.. parsed-literal::

      19	 3.9745750e-01	 2.0664405e-01	 4.3058029e-01	 2.1975864e-01	[ 3.5190881e-01]	 2.0604467e-01
      20	 4.3806310e-01	 2.0096550e-01	 4.7185176e-01	 2.1685296e-01	[ 3.8926652e-01]	 1.8891168e-01


.. parsed-literal::

      21	 4.7978854e-01	 1.9783157e-01	 5.1453405e-01	 2.1513178e-01	[ 4.2783818e-01]	 1.8879867e-01


.. parsed-literal::

      22	 5.6889126e-01	 1.9327857e-01	 6.0522987e-01	 2.1052221e-01	[ 5.0389476e-01]	 2.1505833e-01


.. parsed-literal::

      23	 6.2443204e-01	 1.9236848e-01	 6.6292377e-01	 2.1115213e-01	[ 5.3759512e-01]	 2.0369458e-01
      24	 6.6647717e-01	 1.8954062e-01	 7.0565370e-01	 2.0946648e-01	[ 5.8073509e-01]	 1.9050336e-01


.. parsed-literal::

      25	 6.9361475e-01	 1.8534132e-01	 7.3194461e-01	 2.0471538e-01	[ 6.1695605e-01]	 2.1349978e-01


.. parsed-literal::

      26	 7.2147298e-01	 1.8751016e-01	 7.6105937e-01	 2.0720210e-01	[ 6.3685052e-01]	 2.1081185e-01


.. parsed-literal::

      27	 7.6288627e-01	 1.8479817e-01	 8.0175509e-01	 2.0364716e-01	[ 6.8753357e-01]	 2.1396160e-01


.. parsed-literal::

      28	 8.0339572e-01	 1.8442633e-01	 8.4189799e-01	 2.0188256e-01	[ 7.3714676e-01]	 2.0497036e-01
      29	 8.2753637e-01	 1.9661533e-01	 8.6718910e-01	 2.1126256e-01	[ 7.7414430e-01]	 1.9923806e-01


.. parsed-literal::

      30	 8.6440781e-01	 1.9105035e-01	 9.0527501e-01	 2.0562785e-01	[ 8.0287303e-01]	 2.0037246e-01


.. parsed-literal::

      31	 8.8388896e-01	 1.8671263e-01	 9.2429724e-01	 2.0120768e-01	[ 8.1800739e-01]	 2.1140480e-01


.. parsed-literal::

      32	 9.0509501e-01	 1.8400327e-01	 9.4563273e-01	 1.9743795e-01	[ 8.3558923e-01]	 2.1184421e-01


.. parsed-literal::

      33	 9.2631441e-01	 1.8057171e-01	 9.6803262e-01	 1.9502363e-01	[ 8.5030050e-01]	 2.0285439e-01
      34	 9.3853131e-01	 1.8083048e-01	 9.8063753e-01	 1.9526337e-01	[ 8.6554011e-01]	 1.8655515e-01


.. parsed-literal::

      35	 9.4876166e-01	 1.7906626e-01	 9.9070327e-01	 1.9400989e-01	[ 8.7416620e-01]	 1.8369484e-01


.. parsed-literal::

      36	 9.5928001e-01	 1.7805621e-01	 1.0014169e+00	 1.9390852e-01	[ 8.8558869e-01]	 2.0905042e-01


.. parsed-literal::

      37	 9.7483976e-01	 1.7686677e-01	 1.0177130e+00	 1.9420179e-01	[ 9.0065196e-01]	 2.1421957e-01
      38	 9.8643925e-01	 1.7608204e-01	 1.0312501e+00	 1.9436371e-01	[ 9.1752666e-01]	 1.8603945e-01


.. parsed-literal::

      39	 1.0096631e+00	 1.7391327e-01	 1.0545904e+00	 1.9211099e-01	[ 9.3579249e-01]	 2.1013117e-01


.. parsed-literal::

      40	 1.0198795e+00	 1.7272382e-01	 1.0648949e+00	 1.9101309e-01	[ 9.4106009e-01]	 2.1731687e-01


.. parsed-literal::

      41	 1.0326464e+00	 1.7151671e-01	 1.0781229e+00	 1.8994255e-01	[ 9.4866690e-01]	 2.1743560e-01


.. parsed-literal::

      42	 1.0509678e+00	 1.7014546e-01	 1.0969835e+00	 1.8839943e-01	[ 9.6438855e-01]	 2.1940851e-01


.. parsed-literal::

      43	 1.0605136e+00	 1.6957141e-01	 1.1069253e+00	 1.8748578e-01	[ 9.7557740e-01]	 3.2551098e-01


.. parsed-literal::

      44	 1.0692648e+00	 1.6887571e-01	 1.1155862e+00	 1.8671665e-01	[ 9.8474077e-01]	 2.0649171e-01
      45	 1.0800736e+00	 1.6773665e-01	 1.1265463e+00	 1.8592749e-01	[ 9.9303140e-01]	 2.0278835e-01


.. parsed-literal::

      46	 1.0916949e+00	 1.6627871e-01	 1.1385155e+00	 1.8494201e-01	[ 1.0028743e+00]	 2.1130586e-01


.. parsed-literal::

      47	 1.1018574e+00	 1.6472921e-01	 1.1495660e+00	 1.8449618e-01	[ 1.0101917e+00]	 2.0712805e-01


.. parsed-literal::

      48	 1.1175363e+00	 1.6303795e-01	 1.1649155e+00	 1.8271707e-01	[ 1.0284114e+00]	 2.1313858e-01
      49	 1.1265716e+00	 1.6168224e-01	 1.1740358e+00	 1.8130116e-01	[ 1.0399692e+00]	 1.9479394e-01


.. parsed-literal::

      50	 1.1390854e+00	 1.5982275e-01	 1.1867126e+00	 1.7969534e-01	[ 1.0527913e+00]	 2.1224976e-01


.. parsed-literal::

      51	 1.1406382e+00	 1.5742022e-01	 1.1888863e+00	 1.7629905e-01	[ 1.0665609e+00]	 2.0217657e-01


.. parsed-literal::

      52	 1.1549912e+00	 1.5689097e-01	 1.2026988e+00	 1.7524544e-01	[ 1.0796644e+00]	 2.0369601e-01
      53	 1.1597161e+00	 1.5646210e-01	 1.2073545e+00	 1.7485246e-01	[ 1.0824040e+00]	 1.7874718e-01


.. parsed-literal::

      54	 1.1679467e+00	 1.5537075e-01	 1.2155333e+00	 1.7333393e-01	[ 1.0917922e+00]	 1.7649484e-01
      55	 1.1814011e+00	 1.5347387e-01	 1.2291504e+00	 1.7112375e-01	[ 1.1069127e+00]	 1.8763852e-01


.. parsed-literal::

      56	 1.1880203e+00	 1.5072429e-01	 1.2362888e+00	 1.6655976e-01	[ 1.1130923e+00]	 2.1480656e-01


.. parsed-literal::

      57	 1.2040015e+00	 1.4849940e-01	 1.2521177e+00	 1.6312516e-01	[ 1.1381030e+00]	 2.1408558e-01


.. parsed-literal::

      58	 1.2109213e+00	 1.4921973e-01	 1.2590056e+00	 1.6438817e-01	[ 1.1412271e+00]	 2.1227932e-01


.. parsed-literal::

      59	 1.2191236e+00	 1.4843406e-01	 1.2676910e+00	 1.6330055e-01	  1.1398022e+00 	 2.1011710e-01
      60	 1.2284226e+00	 1.4746253e-01	 1.2774936e+00	 1.6156027e-01	  1.1404795e+00 	 1.7462015e-01


.. parsed-literal::

      61	 1.2375647e+00	 1.4653961e-01	 1.2869486e+00	 1.6002498e-01	[ 1.1470116e+00]	 2.1381831e-01


.. parsed-literal::

      62	 1.2445648e+00	 1.4588677e-01	 1.2941513e+00	 1.5939639e-01	  1.1439054e+00 	 2.0896125e-01
      63	 1.2506012e+00	 1.4543590e-01	 1.3000650e+00	 1.5909255e-01	[ 1.1545677e+00]	 1.7754245e-01


.. parsed-literal::

      64	 1.2613428e+00	 1.4427218e-01	 1.3110572e+00	 1.5784620e-01	[ 1.1698531e+00]	 1.7329216e-01


.. parsed-literal::

      65	 1.2690374e+00	 1.4368576e-01	 1.3190079e+00	 1.5717782e-01	[ 1.1796418e+00]	 2.0840597e-01


.. parsed-literal::

      66	 1.2771622e+00	 1.4279887e-01	 1.3272818e+00	 1.5589271e-01	[ 1.1877848e+00]	 2.0422626e-01


.. parsed-literal::

      67	 1.2849771e+00	 1.4205317e-01	 1.3353675e+00	 1.5484634e-01	[ 1.1898327e+00]	 2.0915365e-01
      68	 1.2957114e+00	 1.4046352e-01	 1.3465575e+00	 1.5307920e-01	  1.1862033e+00 	 1.8273544e-01


.. parsed-literal::

      69	 1.3027171e+00	 1.3952536e-01	 1.3533862e+00	 1.5255573e-01	[ 1.1922312e+00]	 2.1142077e-01


.. parsed-literal::

      70	 1.3096286e+00	 1.3873457e-01	 1.3602410e+00	 1.5178105e-01	[ 1.1950067e+00]	 2.2963238e-01
      71	 1.3160270e+00	 1.3827951e-01	 1.3667003e+00	 1.5189160e-01	[ 1.2036609e+00]	 1.8608379e-01


.. parsed-literal::

      72	 1.3227234e+00	 1.3760439e-01	 1.3736194e+00	 1.5165015e-01	[ 1.2120102e+00]	 2.1848583e-01


.. parsed-literal::

      73	 1.3294890e+00	 1.3749739e-01	 1.3807244e+00	 1.5211734e-01	[ 1.2259530e+00]	 2.1443129e-01


.. parsed-literal::

      74	 1.3361361e+00	 1.3655356e-01	 1.3871926e+00	 1.5087552e-01	[ 1.2325105e+00]	 2.2026944e-01


.. parsed-literal::

      75	 1.3400934e+00	 1.3613997e-01	 1.3910216e+00	 1.5034009e-01	[ 1.2355657e+00]	 2.1151900e-01
      76	 1.3493217e+00	 1.3489713e-01	 1.4005323e+00	 1.4945074e-01	[ 1.2382455e+00]	 1.8233418e-01


.. parsed-literal::

      77	 1.3574851e+00	 1.3388573e-01	 1.4084087e+00	 1.4930521e-01	[ 1.2504924e+00]	 2.1587443e-01


.. parsed-literal::

      78	 1.3647588e+00	 1.3332027e-01	 1.4158162e+00	 1.4904859e-01	[ 1.2553067e+00]	 2.1383953e-01


.. parsed-literal::

      79	 1.3710075e+00	 1.3259148e-01	 1.4224789e+00	 1.4869775e-01	  1.2544108e+00 	 2.0749998e-01
      80	 1.3778146e+00	 1.3181781e-01	 1.4296714e+00	 1.4824260e-01	  1.2550264e+00 	 1.8153954e-01


.. parsed-literal::

      81	 1.3834652e+00	 1.3118456e-01	 1.4353679e+00	 1.4780688e-01	[ 1.2614199e+00]	 2.0534372e-01
      82	 1.3889678e+00	 1.3084320e-01	 1.4408035e+00	 1.4727623e-01	[ 1.2706410e+00]	 1.9239163e-01


.. parsed-literal::

      83	 1.3951203e+00	 1.3046536e-01	 1.4470908e+00	 1.4644815e-01	[ 1.2790538e+00]	 2.0821166e-01
      84	 1.3997894e+00	 1.3030024e-01	 1.4518345e+00	 1.4592513e-01	[ 1.2864460e+00]	 1.7519236e-01


.. parsed-literal::

      85	 1.4039365e+00	 1.3002804e-01	 1.4560269e+00	 1.4604944e-01	[ 1.2897805e+00]	 1.9969940e-01


.. parsed-literal::

      86	 1.4088964e+00	 1.2971192e-01	 1.4612257e+00	 1.4626702e-01	[ 1.2905962e+00]	 2.0944834e-01
      87	 1.4130681e+00	 1.2953707e-01	 1.4654552e+00	 1.4679831e-01	[ 1.2925499e+00]	 1.7838907e-01


.. parsed-literal::

      88	 1.4183935e+00	 1.2925211e-01	 1.4707554e+00	 1.4699517e-01	[ 1.2990677e+00]	 1.9069529e-01


.. parsed-literal::

      89	 1.4226659e+00	 1.2937863e-01	 1.4750017e+00	 1.4738862e-01	  1.2966645e+00 	 2.0958805e-01


.. parsed-literal::

      90	 1.4268346e+00	 1.2910453e-01	 1.4790790e+00	 1.4679134e-01	[ 1.3013442e+00]	 2.1880293e-01


.. parsed-literal::

      91	 1.4303463e+00	 1.2893850e-01	 1.4826203e+00	 1.4634492e-01	[ 1.3033134e+00]	 2.1372056e-01


.. parsed-literal::

      92	 1.4341260e+00	 1.2863716e-01	 1.4864417e+00	 1.4601769e-01	[ 1.3068266e+00]	 2.1009755e-01
      93	 1.4384487e+00	 1.2843022e-01	 1.4907921e+00	 1.4625051e-01	[ 1.3124450e+00]	 2.0517445e-01


.. parsed-literal::

      94	 1.4417155e+00	 1.2822679e-01	 1.4940617e+00	 1.4646784e-01	[ 1.3156433e+00]	 2.0784926e-01


.. parsed-literal::

      95	 1.4475751e+00	 1.2795655e-01	 1.4999980e+00	 1.4688051e-01	[ 1.3237291e+00]	 2.0984960e-01
      96	 1.4488837e+00	 1.2800509e-01	 1.5014341e+00	 1.4758763e-01	  1.3203466e+00 	 1.9920516e-01


.. parsed-literal::

      97	 1.4536525e+00	 1.2786473e-01	 1.5060807e+00	 1.4690969e-01	[ 1.3297308e+00]	 2.1116328e-01


.. parsed-literal::

      98	 1.4559144e+00	 1.2776817e-01	 1.5083706e+00	 1.4654730e-01	[ 1.3327222e+00]	 2.0656896e-01
      99	 1.4582852e+00	 1.2769472e-01	 1.5108211e+00	 1.4632196e-01	[ 1.3338613e+00]	 1.6818810e-01


.. parsed-literal::

     100	 1.4621330e+00	 1.2759377e-01	 1.5147884e+00	 1.4629975e-01	  1.3329049e+00 	 1.8445778e-01
     101	 1.4646385e+00	 1.2769466e-01	 1.5175363e+00	 1.4717180e-01	  1.3305259e+00 	 1.7781878e-01


.. parsed-literal::

     102	 1.4682236e+00	 1.2754943e-01	 1.5209831e+00	 1.4713789e-01	[ 1.3344256e+00]	 1.9395161e-01
     103	 1.4706043e+00	 1.2754955e-01	 1.5233145e+00	 1.4745648e-01	[ 1.3372308e+00]	 1.8107986e-01


.. parsed-literal::

     104	 1.4737243e+00	 1.2758112e-01	 1.5264850e+00	 1.4793596e-01	[ 1.3377150e+00]	 2.1633577e-01


.. parsed-literal::

     105	 1.4758557e+00	 1.2816667e-01	 1.5288225e+00	 1.4915674e-01	  1.3324027e+00 	 2.1418905e-01


.. parsed-literal::

     106	 1.4796501e+00	 1.2794574e-01	 1.5325940e+00	 1.4892795e-01	  1.3349759e+00 	 2.0479536e-01


.. parsed-literal::

     107	 1.4815600e+00	 1.2778024e-01	 1.5345002e+00	 1.4864895e-01	  1.3343720e+00 	 2.1412230e-01


.. parsed-literal::

     108	 1.4841436e+00	 1.2760578e-01	 1.5371748e+00	 1.4833976e-01	  1.3308381e+00 	 2.1390581e-01
     109	 1.4863879e+00	 1.2742876e-01	 1.5396271e+00	 1.4830841e-01	  1.3210667e+00 	 1.8983150e-01


.. parsed-literal::

     110	 1.4895364e+00	 1.2724353e-01	 1.5427210e+00	 1.4786870e-01	  1.3242338e+00 	 1.8673801e-01


.. parsed-literal::

     111	 1.4922946e+00	 1.2708805e-01	 1.5454392e+00	 1.4750130e-01	  1.3281924e+00 	 2.0211411e-01


.. parsed-literal::

     112	 1.4947106e+00	 1.2697770e-01	 1.5478351e+00	 1.4718209e-01	  1.3317067e+00 	 2.1158004e-01


.. parsed-literal::

     113	 1.4991750e+00	 1.2687449e-01	 1.5523222e+00	 1.4666298e-01	  1.3362819e+00 	 2.1831059e-01


.. parsed-literal::

     114	 1.5011042e+00	 1.2678227e-01	 1.5542476e+00	 1.4646290e-01	  1.3365911e+00 	 3.3680749e-01
     115	 1.5038258e+00	 1.2677536e-01	 1.5569813e+00	 1.4633027e-01	[ 1.3397737e+00]	 1.9102907e-01


.. parsed-literal::

     116	 1.5060350e+00	 1.2663464e-01	 1.5592284e+00	 1.4613462e-01	  1.3388364e+00 	 1.8374515e-01


.. parsed-literal::

     117	 1.5082061e+00	 1.2649084e-01	 1.5614903e+00	 1.4582098e-01	  1.3394333e+00 	 2.3526287e-01


.. parsed-literal::

     118	 1.5104527e+00	 1.2630086e-01	 1.5637444e+00	 1.4553975e-01	  1.3396054e+00 	 2.0634174e-01


.. parsed-literal::

     119	 1.5124891e+00	 1.2610709e-01	 1.5657884e+00	 1.4511503e-01	[ 1.3412071e+00]	 2.1773553e-01
     120	 1.5149199e+00	 1.2584781e-01	 1.5682804e+00	 1.4454375e-01	[ 1.3443033e+00]	 1.8623281e-01


.. parsed-literal::

     121	 1.5164828e+00	 1.2543180e-01	 1.5699845e+00	 1.4342455e-01	  1.3384747e+00 	 2.1640968e-01


.. parsed-literal::

     122	 1.5187794e+00	 1.2533675e-01	 1.5722483e+00	 1.4351198e-01	  1.3430633e+00 	 2.0598292e-01


.. parsed-literal::

     123	 1.5202730e+00	 1.2518547e-01	 1.5737579e+00	 1.4350341e-01	  1.3435010e+00 	 2.1829724e-01


.. parsed-literal::

     124	 1.5218703e+00	 1.2496266e-01	 1.5754045e+00	 1.4331579e-01	  1.3411629e+00 	 2.0570230e-01
     125	 1.5245165e+00	 1.2455254e-01	 1.5781801e+00	 1.4279878e-01	  1.3389787e+00 	 1.9840956e-01


.. parsed-literal::

     126	 1.5260500e+00	 1.2422478e-01	 1.5799112e+00	 1.4236395e-01	  1.3281707e+00 	 1.9574213e-01


.. parsed-literal::

     127	 1.5281723e+00	 1.2419073e-01	 1.5819345e+00	 1.4218797e-01	  1.3339414e+00 	 2.1139860e-01


.. parsed-literal::

     128	 1.5294088e+00	 1.2410857e-01	 1.5831790e+00	 1.4195607e-01	  1.3365438e+00 	 2.0878124e-01


.. parsed-literal::

     129	 1.5311985e+00	 1.2393167e-01	 1.5850338e+00	 1.4172788e-01	  1.3387161e+00 	 2.0982623e-01
     130	 1.5320669e+00	 1.2353947e-01	 1.5861276e+00	 1.4100292e-01	  1.3390597e+00 	 2.0583892e-01


.. parsed-literal::

     131	 1.5347329e+00	 1.2346841e-01	 1.5887046e+00	 1.4123019e-01	  1.3401920e+00 	 2.0686865e-01


.. parsed-literal::

     132	 1.5356647e+00	 1.2338413e-01	 1.5896282e+00	 1.4129311e-01	  1.3397206e+00 	 2.0376563e-01


.. parsed-literal::

     133	 1.5371005e+00	 1.2323660e-01	 1.5910649e+00	 1.4134696e-01	  1.3396510e+00 	 2.1268177e-01
     134	 1.5391508e+00	 1.2307279e-01	 1.5931483e+00	 1.4139492e-01	  1.3390787e+00 	 1.7911077e-01


.. parsed-literal::

     135	 1.5403506e+00	 1.2300362e-01	 1.5943641e+00	 1.4152423e-01	  1.3393973e+00 	 3.2206059e-01


.. parsed-literal::

     136	 1.5416848e+00	 1.2297999e-01	 1.5957117e+00	 1.4146629e-01	  1.3402019e+00 	 2.0237446e-01
     137	 1.5427631e+00	 1.2299747e-01	 1.5968004e+00	 1.4141558e-01	  1.3402662e+00 	 1.8122983e-01


.. parsed-literal::

     138	 1.5442812e+00	 1.2299474e-01	 1.5983358e+00	 1.4140817e-01	  1.3386365e+00 	 1.7909670e-01


.. parsed-literal::

     139	 1.5457067e+00	 1.2300928e-01	 1.5998353e+00	 1.4167495e-01	  1.3355023e+00 	 2.0860052e-01


.. parsed-literal::

     140	 1.5483132e+00	 1.2294988e-01	 1.6024144e+00	 1.4177202e-01	  1.3295236e+00 	 2.1257162e-01


.. parsed-literal::

     141	 1.5494041e+00	 1.2283746e-01	 1.6034769e+00	 1.4180874e-01	  1.3278085e+00 	 2.0390773e-01


.. parsed-literal::

     142	 1.5506963e+00	 1.2271506e-01	 1.6047579e+00	 1.4200810e-01	  1.3248124e+00 	 2.0188665e-01
     143	 1.5518416e+00	 1.2262376e-01	 1.6059681e+00	 1.4238444e-01	  1.3184493e+00 	 2.0157051e-01


.. parsed-literal::

     144	 1.5534718e+00	 1.2261089e-01	 1.6075780e+00	 1.4264854e-01	  1.3192968e+00 	 1.8373299e-01


.. parsed-literal::

     145	 1.5549786e+00	 1.2265342e-01	 1.6091121e+00	 1.4301664e-01	  1.3217593e+00 	 2.0954609e-01


.. parsed-literal::

     146	 1.5560555e+00	 1.2266331e-01	 1.6102269e+00	 1.4325668e-01	  1.3229415e+00 	 2.1813416e-01


.. parsed-literal::

     147	 1.5584205e+00	 1.2251078e-01	 1.6126774e+00	 1.4357706e-01	  1.3248716e+00 	 2.1876073e-01


.. parsed-literal::

     148	 1.5591913e+00	 1.2246424e-01	 1.6136358e+00	 1.4431752e-01	  1.3193521e+00 	 2.1561503e-01


.. parsed-literal::

     149	 1.5611347e+00	 1.2229739e-01	 1.6154471e+00	 1.4386032e-01	  1.3232495e+00 	 2.1709323e-01
     150	 1.5619737e+00	 1.2218344e-01	 1.6162546e+00	 1.4370525e-01	  1.3244862e+00 	 1.9372845e-01


.. parsed-literal::

    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 4s, sys: 1.09 s, total: 2min 6s
    Wall time: 31.6 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f16949e4b50>



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
    CPU times: user 2.07 s, sys: 39.9 ms, total: 2.11 s
    Wall time: 638 ms


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

