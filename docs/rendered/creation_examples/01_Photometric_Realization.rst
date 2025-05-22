Photometric Realization from Different Magnitude Error Models
=============================================================

author: John Franklin Crenshaw, Sam Schmidt, Eric Charles, Ziang Yan

last run successfully: August 2, 2023

This notebook demonstrates how to do photometric realization from
different magnitude error models. For more completed degrader demo, see
``degradation-demo.ipynb``

.. code:: ipython3

    import matplotlib.pyplot as plt
    from pzflow.examples import get_example_flow
    from rail.creation.engines.flowEngine import FlowCreator
    from rail.creation.degraders.photometric_errors import LSSTErrorModel
    from rail.core.stage import RailStage


Specify the path to the pretrained ‘pzflow’ used to generate samples

.. code:: ipython3

    import pzflow
    import os
    
    flow_file = os.path.join(
        os.path.dirname(pzflow.__file__), "example_files", "example-flow.pzflow.pkl"
    )


We’ll start by setting up the RAIL data store. RAIL uses
`ceci <https://github.com/LSSTDESC/ceci>`__, which is designed for
pipelines rather than interactive notebooks, the data store will work
around that and enable us to use data interactively. See the
``rail/examples/goldenspike_examples/goldenspike.ipynb`` example
notebook for more details on the Data Store.

.. code:: ipython3

    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True


“True” Engine
~~~~~~~~~~~~~

First, let’s make an Engine that has no degradation. We can use it to
generate a “true” sample, to which we can compare all the degraded
samples below.

Note: in this example, we will use a normalizing flow engine from the
`pzflow <https://github.com/jfcrenshaw/pzflow>`__ package. However,
everything in this notebook is totally agnostic to what the underlying
engine is.

The Engine is a type of RailStage object, so we can make one using the
``RailStage.make_stage`` function for the class of Engine that we want.
We then pass in the configuration parameters as arguments to
``make_stage``.

.. code:: ipython3

    n_samples = int(1e5)
    flowEngine_truth = FlowCreator.make_stage(
        name="truth", model=flow_file, n_samples=n_samples
    )



.. parsed-literal::

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.17/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7fbbf4685660>



Now we invoke the ``sample`` method to generate some samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that this will return a ``DataHandle`` object, which can keep both
the data itself, and also the path to where the data is written. When
talking to rail stages we can use this as though it were the underlying
data and pass it as an argument. This allows the rail stages to keep
track of where their inputs are coming from.

To calculate magnitude error for extended sources, we need the
information about major and minor axes of each galaxy. Here we simply
generate random values

.. code:: ipython3

    samples_truth = flowEngine_truth.sample(n_samples, seed=0)
    
    import numpy as np
    
    samples_truth.data["major"] = np.abs(
        np.random.normal(loc=0.01, scale=0.1, size=n_samples)
    )  # add major and minor axes
    b_to_a = 1 - 0.5 * np.random.rand(n_samples)
    samples_truth.data["minor"] = samples_truth.data["major"] * b_to_a
    
    print(samples_truth())
    print("Data was written to ", samples_truth.path)



.. parsed-literal::

    Inserting handle into data store.  output_truth: inprogress_output_truth.pq, truth
           redshift          u          g          r          i          z  \
    0      1.398944  27.667536  26.723337  26.032637  25.178587  24.695955   
    1      2.285624  28.786999  27.476589  26.640175  26.259745  25.865673   
    2      1.495132  30.011349  29.789337  28.200390  26.014826  25.030174   
    3      0.842594  29.306244  28.721798  27.353018  26.256907  25.529823   
    4      1.588960  26.273870  26.115387  25.950441  25.687405  25.466606   
    ...         ...        ...        ...        ...        ...        ...   
    99995  0.389450  27.270800  26.371506  25.436853  25.077412  24.852779   
    99996  1.481047  27.478113  26.735254  26.042776  25.204935  24.825092   
    99997  2.023548  26.990147  26.714737  26.377949  26.250343  25.917370   
    99998  1.548204  26.367432  26.206884  26.087980  25.876932  25.715893   
    99999  1.739491  26.881983  26.773064  26.553123  26.319622  25.955982   
    
                   y     major     minor  
    0      23.994413  0.019541  0.009875  
    1      25.391064  0.101342  0.101101  
    2      24.304707  0.136188  0.106744  
    3      25.291103  0.043209  0.021739  
    4      25.096743  0.227012  0.148440  
    ...          ...       ...       ...  
    99995  24.737946  0.033692  0.031004  
    99996  24.224169  0.166796  0.083829  
    99997  25.613836  0.069803  0.065824  
    99998  25.274899  0.015296  0.011167  
    99999  25.699642  0.086404  0.052639  
    
    [100000 rows x 9 columns]
    Data was written to  output_truth.pq


LSSTErrorModel
~~~~~~~~~~~~~~

Now, we will demonstrate the ``LSSTErrorModel``, which adds photometric
errors using a model similar to the model from `Ivezic et
al. 2019 <https://arxiv.org/abs/0805.2366>`__ (specifically, it uses the
model from this paper, without making the high SNR assumption. To
restore this assumption and therefore use the exact model from the
paper, set ``highSNR=True``.)

Let’s create an error model with the default settings for point sources:

.. code:: ipython3

    errorModel = LSSTErrorModel.make_stage(name="error_model")


For extended sources:

.. code:: ipython3

    errorModel_auto = LSSTErrorModel.make_stage(
        name="error_model_auto", extendedType="auto"
    )


.. code:: ipython3

    errorModel_gaap = LSSTErrorModel.make_stage(
        name="error_model_gaap", extendedType="gaap"
    )


Now let’s add this error model as a degrader and draw some samples with
photometric errors.

.. code:: ipython3

    samples_w_errs = errorModel(samples_truth)
    samples_w_errs()



.. parsed-literal::

    Inserting handle into data store.  output_error_model: inprogress_output_error_model.pq, error_model




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398944</td>
          <td>27.290094</td>
          <td>0.673093</td>
          <td>27.023277</td>
          <td>0.215922</td>
          <td>25.932831</td>
          <td>0.074304</td>
          <td>25.206960</td>
          <td>0.063789</td>
          <td>24.747588</td>
          <td>0.081259</td>
          <td>24.043261</td>
          <td>0.098168</td>
          <td>0.019541</td>
          <td>0.009875</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.203080</td>
          <td>0.250566</td>
          <td>26.756152</td>
          <td>0.152442</td>
          <td>26.121329</td>
          <td>0.142162</td>
          <td>25.688601</td>
          <td>0.183698</td>
          <td>24.836941</td>
          <td>0.194481</td>
          <td>0.101342</td>
          <td>0.101101</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>31.181963</td>
          <td>3.760366</td>
          <td>30.503974</td>
          <td>2.018997</td>
          <td>28.237782</td>
          <td>0.502966</td>
          <td>26.044336</td>
          <td>0.133024</td>
          <td>24.845128</td>
          <td>0.088550</td>
          <td>24.185603</td>
          <td>0.111181</td>
          <td>0.136188</td>
          <td>0.106744</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.760093</td>
          <td>1.429976</td>
          <td>27.785043</td>
          <td>0.356285</td>
          <td>26.108585</td>
          <td>0.140610</td>
          <td>25.587136</td>
          <td>0.168541</td>
          <td>25.222509</td>
          <td>0.267756</td>
          <td>0.043209</td>
          <td>0.021739</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>27.128017</td>
          <td>0.601261</td>
          <td>26.123493</td>
          <td>0.099849</td>
          <td>25.904017</td>
          <td>0.072434</td>
          <td>25.738387</td>
          <td>0.101927</td>
          <td>25.488632</td>
          <td>0.154943</td>
          <td>24.747357</td>
          <td>0.180310</td>
          <td>0.227012</td>
          <td>0.148440</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>27.743407</td>
          <td>0.905972</td>
          <td>26.227641</td>
          <td>0.109360</td>
          <td>25.483043</td>
          <td>0.049866</td>
          <td>25.098563</td>
          <td>0.057941</td>
          <td>24.843724</td>
          <td>0.088441</td>
          <td>25.184356</td>
          <td>0.259539</td>
          <td>0.033692</td>
          <td>0.031004</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.718560</td>
          <td>0.445719</td>
          <td>26.471073</td>
          <td>0.135079</td>
          <td>26.081052</td>
          <td>0.084689</td>
          <td>25.196205</td>
          <td>0.063184</td>
          <td>24.879767</td>
          <td>0.091289</td>
          <td>24.207585</td>
          <td>0.113333</td>
          <td>0.166796</td>
          <td>0.083829</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.043479</td>
          <td>0.566146</td>
          <td>26.462335</td>
          <td>0.134064</td>
          <td>26.428166</td>
          <td>0.114803</td>
          <td>26.233155</td>
          <td>0.156489</td>
          <td>26.018914</td>
          <td>0.242095</td>
          <td>25.331077</td>
          <td>0.292405</td>
          <td>0.069803</td>
          <td>0.065824</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.315149</td>
          <td>0.326069</td>
          <td>26.144215</td>
          <td>0.101676</td>
          <td>25.973885</td>
          <td>0.077049</td>
          <td>25.887426</td>
          <td>0.116093</td>
          <td>25.453319</td>
          <td>0.150322</td>
          <td>26.174948</td>
          <td>0.558134</td>
          <td>0.015296</td>
          <td>0.011167</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.246641</td>
          <td>0.653248</td>
          <td>26.722605</td>
          <td>0.167588</td>
          <td>26.749288</td>
          <td>0.151548</td>
          <td>26.074165</td>
          <td>0.136497</td>
          <td>25.782052</td>
          <td>0.198758</td>
          <td>26.370959</td>
          <td>0.641189</td>
          <td>0.086404</td>
          <td>0.052639</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_gaap = errorModel_gaap(samples_truth)
    samples_w_errs_gaap.data



.. parsed-literal::

    Inserting handle into data store.  output_error_model_gaap: inprogress_output_error_model_gaap.pq, error_model_gaap




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398944</td>
          <td>26.752441</td>
          <td>0.505180</td>
          <td>26.652198</td>
          <td>0.181374</td>
          <td>25.972703</td>
          <td>0.090598</td>
          <td>25.104664</td>
          <td>0.069125</td>
          <td>24.722760</td>
          <td>0.093545</td>
          <td>23.964456</td>
          <td>0.108303</td>
          <td>0.019541</td>
          <td>0.009875</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.958016</td>
          <td>1.135173</td>
          <td>29.100478</td>
          <td>1.111950</td>
          <td>26.390427</td>
          <td>0.134737</td>
          <td>26.352810</td>
          <td>0.210615</td>
          <td>25.675601</td>
          <td>0.218923</td>
          <td>25.113024</td>
          <td>0.295059</td>
          <td>0.101342</td>
          <td>0.101101</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>29.460348</td>
          <td>2.316674</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.173962</td>
          <td>0.571093</td>
          <td>25.904906</td>
          <td>0.146153</td>
          <td>25.169988</td>
          <td>0.144671</td>
          <td>24.576225</td>
          <td>0.192160</td>
          <td>0.136188</td>
          <td>0.106744</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.847705</td>
          <td>0.542630</td>
          <td>27.971842</td>
          <td>0.519650</td>
          <td>27.059150</td>
          <td>0.230766</td>
          <td>26.287004</td>
          <td>0.193532</td>
          <td>25.310353</td>
          <td>0.156264</td>
          <td>25.022364</td>
          <td>0.266532</td>
          <td>0.043209</td>
          <td>0.021739</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.953800</td>
          <td>0.293866</td>
          <td>25.983461</td>
          <td>0.112532</td>
          <td>26.061054</td>
          <td>0.109124</td>
          <td>25.769198</td>
          <td>0.138338</td>
          <td>25.544810</td>
          <td>0.210965</td>
          <td>25.396245</td>
          <td>0.395619</td>
          <td>0.227012</td>
          <td>0.148440</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>27.152826</td>
          <td>0.672743</td>
          <td>26.538699</td>
          <td>0.165107</td>
          <td>25.435989</td>
          <td>0.056532</td>
          <td>25.155376</td>
          <td>0.072504</td>
          <td>24.927276</td>
          <td>0.112180</td>
          <td>25.478297</td>
          <td>0.383089</td>
          <td>0.033692</td>
          <td>0.031004</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.595620</td>
          <td>0.181281</td>
          <td>25.975493</td>
          <td>0.095876</td>
          <td>25.294593</td>
          <td>0.086474</td>
          <td>24.874090</td>
          <td>0.112701</td>
          <td>24.138373</td>
          <td>0.133081</td>
          <td>0.166796</td>
          <td>0.083829</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.853614</td>
          <td>0.548890</td>
          <td>26.854483</td>
          <td>0.217679</td>
          <td>26.421368</td>
          <td>0.135920</td>
          <td>26.076129</td>
          <td>0.163714</td>
          <td>25.743447</td>
          <td>0.227659</td>
          <td>25.910496</td>
          <td>0.535744</td>
          <td>0.069803</td>
          <td>0.065824</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.894703</td>
          <td>0.560168</td>
          <td>26.257121</td>
          <td>0.129314</td>
          <td>26.426116</td>
          <td>0.134504</td>
          <td>26.054898</td>
          <td>0.158391</td>
          <td>26.067025</td>
          <td>0.292629</td>
          <td>25.060545</td>
          <td>0.274073</td>
          <td>0.015296</td>
          <td>0.011167</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.566162</td>
          <td>0.444355</td>
          <td>26.695489</td>
          <td>0.190802</td>
          <td>26.272050</td>
          <td>0.119623</td>
          <td>26.157641</td>
          <td>0.175760</td>
          <td>26.545470</td>
          <td>0.432102</td>
          <td>25.660624</td>
          <td>0.445817</td>
          <td>0.086404</td>
          <td>0.052639</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_auto = errorModel_auto(samples_truth)
    samples_w_errs_auto.data



.. parsed-literal::

    Inserting handle into data store.  output_error_model_auto: inprogress_output_error_model_auto.pq, error_model_auto




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398944</td>
          <td>29.394443</td>
          <td>2.108945</td>
          <td>26.676212</td>
          <td>0.161531</td>
          <td>26.024824</td>
          <td>0.080856</td>
          <td>25.266514</td>
          <td>0.067478</td>
          <td>24.658360</td>
          <td>0.075349</td>
          <td>23.904860</td>
          <td>0.087223</td>
          <td>0.019541</td>
          <td>0.009875</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.309467</td>
          <td>0.729763</td>
          <td>27.565975</td>
          <td>0.369715</td>
          <td>26.637742</td>
          <td>0.155173</td>
          <td>26.588853</td>
          <td>0.238576</td>
          <td>25.823143</td>
          <td>0.230982</td>
          <td>24.912060</td>
          <td>0.233511</td>
          <td>0.101342</td>
          <td>0.101101</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.059802</td>
          <td>0.160045</td>
          <td>24.863740</td>
          <td>0.106472</td>
          <td>24.207645</td>
          <td>0.134584</td>
          <td>0.136188</td>
          <td>0.106744</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.113081</td>
          <td>0.600153</td>
          <td>28.524624</td>
          <td>0.690244</td>
          <td>27.362658</td>
          <td>0.257480</td>
          <td>26.283014</td>
          <td>0.165910</td>
          <td>25.390188</td>
          <td>0.144566</td>
          <td>25.557762</td>
          <td>0.355396</td>
          <td>0.043209</td>
          <td>0.021739</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.031292</td>
          <td>0.327295</td>
          <td>25.967755</td>
          <td>0.117422</td>
          <td>25.849947</td>
          <td>0.096384</td>
          <td>25.473690</td>
          <td>0.113849</td>
          <td>25.649501</td>
          <td>0.243681</td>
          <td>25.401185</td>
          <td>0.419127</td>
          <td>0.227012</td>
          <td>0.148440</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>26.784338</td>
          <td>0.472182</td>
          <td>26.287971</td>
          <td>0.116661</td>
          <td>25.425530</td>
          <td>0.048062</td>
          <td>25.138444</td>
          <td>0.060930</td>
          <td>24.796552</td>
          <td>0.086046</td>
          <td>25.027643</td>
          <td>0.231254</td>
          <td>0.033692</td>
          <td>0.031004</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>25.978397</td>
          <td>0.281655</td>
          <td>26.856052</td>
          <td>0.219237</td>
          <td>26.005965</td>
          <td>0.095218</td>
          <td>25.119705</td>
          <td>0.071542</td>
          <td>24.703088</td>
          <td>0.093849</td>
          <td>23.953245</td>
          <td>0.109489</td>
          <td>0.166796</td>
          <td>0.083829</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.532298</td>
          <td>0.149736</td>
          <td>26.501028</td>
          <td>0.129622</td>
          <td>26.242046</td>
          <td>0.167406</td>
          <td>25.875356</td>
          <td>0.227201</td>
          <td>26.319998</td>
          <td>0.649419</td>
          <td>0.069803</td>
          <td>0.065824</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.909162</td>
          <td>0.235007</td>
          <td>26.147194</td>
          <td>0.102155</td>
          <td>26.092728</td>
          <td>0.085774</td>
          <td>26.024340</td>
          <td>0.131073</td>
          <td>26.097224</td>
          <td>0.258772</td>
          <td>25.992715</td>
          <td>0.489587</td>
          <td>0.015296</td>
          <td>0.011167</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.276716</td>
          <td>0.329532</td>
          <td>26.760335</td>
          <td>0.182689</td>
          <td>26.366792</td>
          <td>0.115976</td>
          <td>26.078558</td>
          <td>0.146337</td>
          <td>25.864255</td>
          <td>0.226258</td>
          <td>25.168093</td>
          <td>0.272425</td>
          <td>0.086404</td>
          <td>0.052639</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



Notice some of the magnitudes are inf’s. These are non-detections
(i.e. the noisy flux was negative). You can change the nSigma limit for
non-detections by setting ``sigLim=...``. For example, if ``sigLim=5``,
then all fluxes with ``SNR<5`` are flagged as non-detections.

Let’s plot the error as a function of magnitude

.. code:: ipython3

    %matplotlib inline
    
    fig, axes_ = plt.subplots(ncols=3, nrows=2, figsize=(15, 9), dpi=100)
    axes = axes_.reshape(-1)
    for i, band in enumerate("ugrizy"):
        ax = axes[i]
        # pull out the magnitudes and errors
        mags = samples_w_errs.data[band].to_numpy()
        errs = samples_w_errs.data[band + "_err"].to_numpy()
        
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
        
        # plot errs vs mags
        #ax.plot(mags, errs, label=band) 
        
        #plt.plot(mags, errs, c='C'+str(i))
        ax.scatter(samples_w_errs_gaap.data[band].to_numpy(),
                samples_w_errs_gaap.data[band + "_err"].to_numpy(),
                    s=5, marker='.', color='C0', alpha=0.8, label='GAAP')
        
        ax.plot(mags, errs, color='C3', label='Point source')
        
        
        ax.legend()
        ax.set_xlim(18, 31)
        ax.set_ylim(-0.1, 3.5)
        ax.set(xlabel=band+" Band Magnitude (AB)", ylabel="Error (mags)")




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_24_0.png


.. code:: ipython3

    %matplotlib inline
    
    fig, axes_ = plt.subplots(ncols=3, nrows=2, figsize=(15, 9), dpi=100)
    axes = axes_.reshape(-1)
    for i, band in enumerate("ugrizy"):
        ax = axes[i]
        # pull out the magnitudes and errors
        mags = samples_w_errs.data[band].to_numpy()
        errs = samples_w_errs.data[band + "_err"].to_numpy()
        
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
        
        # plot errs vs mags
        #ax.plot(mags, errs, label=band) 
        
        #plt.plot(mags, errs, c='C'+str(i))
        ax.scatter(samples_w_errs_auto.data[band].to_numpy(),
                samples_w_errs_auto.data[band + "_err"].to_numpy(),
                    s=5, marker='.', color='C0', alpha=0.8, label='AUTO')
        
        ax.plot(mags, errs, color='C3', label='Point source')
        
        
        ax.legend()
        ax.set_xlim(18, 31)
        ax.set_ylim(-0.1, 3.5)
        ax.set(xlabel=band+" Band Magnitude (AB)", ylabel="Error (mags)")




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_25_0.png


You can see that the photometric error increases as magnitude gets
dimmer, just like you would expect, and that the extended source errors
are greater than the point source errors. The extended source errors are
also scattered, because the galaxies have random sizes.

Also, you can find the GAaP and AUTO magnitude error are scattered due
to variable galaxy sizes. Also, you can find that there are gaps between
GAAP magnitude error and point souce magnitude error, this is because
the additional factors due to aperture sizes have a minimum value of
:math:`\sqrt{(\sigma^2+A_{\mathrm{min}})/\sigma^2}`, where
:math:`\sigma` is the width of the beam, :math:`A_{\min}` is an offset
of the aperture sizes (taken to be 0.7 arcmin here).

You can also see that there are *very* faint galaxies in this sample.
That’s because, by default, the error model returns magnitudes for all
positive fluxes. If you want these galaxies flagged as non-detections
instead, you can set e.g. ``sigLim=5``, and everything with ``SNR<5``
will be flagged as a non-detection.
