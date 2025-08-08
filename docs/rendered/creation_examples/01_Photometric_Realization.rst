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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f8851d69750>



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
    0      23.994413  0.070438  0.048273  
    1      25.391064  0.049082  0.035795  
    2      24.304707  0.058784  0.049343  
    3      25.291103  0.031404  0.018829  
    4      25.096743  0.192219  0.141252  
    ...          ...       ...       ...  
    99995  24.737946  0.044785  0.042718  
    99996  24.224169  0.018655  0.011994  
    99997  25.613836  0.071890  0.047096  
    99998  25.274899  0.002237  0.001409  
    99999  25.699642  0.008095  0.007704  
    
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

    Inserting handle into data store.  input: None, error_model
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
          <td>28.440801</td>
          <td>1.353083</td>
          <td>26.738486</td>
          <td>0.169868</td>
          <td>26.077389</td>
          <td>0.084417</td>
          <td>25.152409</td>
          <td>0.060776</td>
          <td>24.616344</td>
          <td>0.072364</td>
          <td>24.036379</td>
          <td>0.097578</td>
          <td>0.070438</td>
          <td>0.048273</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.747473</td>
          <td>0.455527</td>
          <td>27.335460</td>
          <td>0.279158</td>
          <td>26.675509</td>
          <td>0.142236</td>
          <td>26.144677</td>
          <td>0.145048</td>
          <td>26.053826</td>
          <td>0.249155</td>
          <td>24.907962</td>
          <td>0.206434</td>
          <td>0.049082</td>
          <td>0.035795</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.797015</td>
          <td>0.359646</td>
          <td>25.948915</td>
          <td>0.122468</td>
          <td>25.083548</td>
          <td>0.109133</td>
          <td>24.268054</td>
          <td>0.119458</td>
          <td>0.058784</td>
          <td>0.049343</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.270256</td>
          <td>0.663980</td>
          <td>27.798057</td>
          <td>0.402483</td>
          <td>27.284077</td>
          <td>0.237872</td>
          <td>26.713868</td>
          <td>0.234625</td>
          <td>25.683112</td>
          <td>0.182847</td>
          <td>25.355411</td>
          <td>0.298195</td>
          <td>0.031404</td>
          <td>0.018829</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.938944</td>
          <td>0.240469</td>
          <td>26.179711</td>
          <td>0.104880</td>
          <td>26.109611</td>
          <td>0.086847</td>
          <td>25.536350</td>
          <td>0.085353</td>
          <td>25.332201</td>
          <td>0.135437</td>
          <td>24.771569</td>
          <td>0.184043</td>
          <td>0.192219</td>
          <td>0.141252</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.424693</td>
          <td>0.129774</td>
          <td>25.458977</td>
          <td>0.048811</td>
          <td>25.090249</td>
          <td>0.057515</td>
          <td>24.906765</td>
          <td>0.093481</td>
          <td>24.532952</td>
          <td>0.150179</td>
          <td>0.044785</td>
          <td>0.042718</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.028662</td>
          <td>0.560156</td>
          <td>26.683864</td>
          <td>0.162145</td>
          <td>26.016217</td>
          <td>0.079983</td>
          <td>25.175579</td>
          <td>0.062038</td>
          <td>24.891961</td>
          <td>0.092273</td>
          <td>24.314069</td>
          <td>0.124328</td>
          <td>0.018655</td>
          <td>0.011994</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.295647</td>
          <td>0.675660</td>
          <td>26.689870</td>
          <td>0.162978</td>
          <td>26.368510</td>
          <td>0.108983</td>
          <td>26.127732</td>
          <td>0.142948</td>
          <td>25.892983</td>
          <td>0.218094</td>
          <td>25.344192</td>
          <td>0.295513</td>
          <td>0.071890</td>
          <td>0.047096</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.596613</td>
          <td>0.406225</td>
          <td>26.212555</td>
          <td>0.107930</td>
          <td>25.961559</td>
          <td>0.076215</td>
          <td>25.918924</td>
          <td>0.119318</td>
          <td>25.764468</td>
          <td>0.195841</td>
          <td>25.469969</td>
          <td>0.326806</td>
          <td>0.002237</td>
          <td>0.001409</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.504775</td>
          <td>0.139063</td>
          <td>26.467895</td>
          <td>0.118842</td>
          <td>26.063357</td>
          <td>0.135229</td>
          <td>25.968613</td>
          <td>0.232237</td>
          <td>25.376342</td>
          <td>0.303254</td>
          <td>0.008095</td>
          <td>0.007704</td>
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
          <td>26.809326</td>
          <td>0.530482</td>
          <td>26.379992</td>
          <td>0.145251</td>
          <td>26.008275</td>
          <td>0.094549</td>
          <td>25.194540</td>
          <td>0.075740</td>
          <td>24.786031</td>
          <td>0.100017</td>
          <td>23.888764</td>
          <td>0.102564</td>
          <td>0.070438</td>
          <td>0.048273</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.537356</td>
          <td>0.431684</td>
          <td>27.680756</td>
          <td>0.418727</td>
          <td>26.784922</td>
          <td>0.183814</td>
          <td>26.457575</td>
          <td>0.223726</td>
          <td>26.467517</td>
          <td>0.403304</td>
          <td>24.823378</td>
          <td>0.226764</td>
          <td>0.049082</td>
          <td>0.035795</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.607657</td>
          <td>1.581646</td>
          <td>30.559423</td>
          <td>2.214525</td>
          <td>28.516865</td>
          <td>0.703533</td>
          <td>26.000794</td>
          <td>0.152651</td>
          <td>25.141878</td>
          <td>0.135988</td>
          <td>24.382269</td>
          <td>0.156881</td>
          <td>0.058784</td>
          <td>0.049343</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.968550</td>
          <td>1.124289</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.803967</td>
          <td>0.417667</td>
          <td>26.310579</td>
          <td>0.197079</td>
          <td>25.715842</td>
          <td>0.219745</td>
          <td>24.902057</td>
          <td>0.241092</td>
          <td>0.031404</td>
          <td>0.018829</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.573362</td>
          <td>0.468289</td>
          <td>26.186430</td>
          <td>0.131523</td>
          <td>26.001994</td>
          <td>0.101390</td>
          <td>25.663939</td>
          <td>0.123520</td>
          <td>26.354441</td>
          <td>0.396987</td>
          <td>25.231967</td>
          <td>0.341127</td>
          <td>0.192219</td>
          <td>0.141252</td>
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
          <td>27.881472</td>
          <td>1.071236</td>
          <td>26.233124</td>
          <td>0.127320</td>
          <td>25.502129</td>
          <td>0.060125</td>
          <td>25.004010</td>
          <td>0.063606</td>
          <td>24.939746</td>
          <td>0.113735</td>
          <td>24.514907</td>
          <td>0.175053</td>
          <td>0.044785</td>
          <td>0.042718</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.878147</td>
          <td>0.219254</td>
          <td>25.987211</td>
          <td>0.091763</td>
          <td>25.275257</td>
          <td>0.080372</td>
          <td>24.762657</td>
          <td>0.096880</td>
          <td>24.105743</td>
          <td>0.122480</td>
          <td>0.018655</td>
          <td>0.011994</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.187509</td>
          <td>1.276459</td>
          <td>26.910993</td>
          <td>0.227556</td>
          <td>26.469736</td>
          <td>0.141287</td>
          <td>26.255111</td>
          <td>0.189983</td>
          <td>25.665149</td>
          <td>0.212681</td>
          <td>26.112022</td>
          <td>0.617163</td>
          <td>0.071890</td>
          <td>0.047096</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.676687</td>
          <td>0.216194</td>
          <td>26.307349</td>
          <td>0.134975</td>
          <td>26.162175</td>
          <td>0.106876</td>
          <td>25.850389</td>
          <td>0.132769</td>
          <td>26.359818</td>
          <td>0.368992</td>
          <td>24.548984</td>
          <td>0.179033</td>
          <td>0.002237</td>
          <td>0.001409</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.212719</td>
          <td>0.334268</td>
          <td>26.776320</td>
          <td>0.201263</td>
          <td>26.456576</td>
          <td>0.138033</td>
          <td>26.347496</td>
          <td>0.202870</td>
          <td>25.704363</td>
          <td>0.217228</td>
          <td>26.292049</td>
          <td>0.692146</td>
          <td>0.008095</td>
          <td>0.007704</td>
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
          <td>30.059552</td>
          <td>2.732326</td>
          <td>26.700396</td>
          <td>0.171102</td>
          <td>25.945055</td>
          <td>0.078741</td>
          <td>25.139190</td>
          <td>0.063128</td>
          <td>24.758098</td>
          <td>0.085965</td>
          <td>23.910637</td>
          <td>0.091736</td>
          <td>0.070438</td>
          <td>0.048273</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.640367</td>
          <td>0.426294</td>
          <td>28.027017</td>
          <td>0.487299</td>
          <td>26.875746</td>
          <td>0.172878</td>
          <td>26.516667</td>
          <td>0.203968</td>
          <td>25.720507</td>
          <td>0.193182</td>
          <td>25.552243</td>
          <td>0.356769</td>
          <td>0.049082</td>
          <td>0.035795</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>32.290722</td>
          <td>4.877364</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.228674</td>
          <td>1.711408</td>
          <td>26.008528</td>
          <td>0.134182</td>
          <td>25.145947</td>
          <td>0.119699</td>
          <td>24.448524</td>
          <td>0.145225</td>
          <td>0.058784</td>
          <td>0.049343</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.789668</td>
          <td>0.819528</td>
          <td>27.978320</td>
          <td>0.417130</td>
          <td>26.130027</td>
          <td>0.144564</td>
          <td>25.559400</td>
          <td>0.166048</td>
          <td>24.701167</td>
          <td>0.174957</td>
          <td>0.031404</td>
          <td>0.018829</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.519414</td>
          <td>0.459174</td>
          <td>26.157970</td>
          <td>0.131873</td>
          <td>26.059433</td>
          <td>0.109820</td>
          <td>25.674668</td>
          <td>0.128491</td>
          <td>25.210995</td>
          <td>0.160300</td>
          <td>26.058953</td>
          <td>0.647722</td>
          <td>0.192219</td>
          <td>0.141252</td>
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
          <td>26.991668</td>
          <td>0.553393</td>
          <td>26.497072</td>
          <td>0.141164</td>
          <td>25.387346</td>
          <td>0.046996</td>
          <td>25.053565</td>
          <td>0.057192</td>
          <td>24.899501</td>
          <td>0.095269</td>
          <td>24.997432</td>
          <td>0.228045</td>
          <td>0.044785</td>
          <td>0.042718</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.335531</td>
          <td>0.695545</td>
          <td>26.658729</td>
          <td>0.159149</td>
          <td>26.146455</td>
          <td>0.090009</td>
          <td>25.404036</td>
          <td>0.076215</td>
          <td>24.789472</td>
          <td>0.084598</td>
          <td>24.203265</td>
          <td>0.113295</td>
          <td>0.018655</td>
          <td>0.011994</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.568288</td>
          <td>0.830521</td>
          <td>26.675612</td>
          <td>0.167607</td>
          <td>26.289715</td>
          <td>0.106646</td>
          <td>26.199761</td>
          <td>0.159614</td>
          <td>25.754741</td>
          <td>0.203240</td>
          <td>25.088330</td>
          <td>0.251147</td>
          <td>0.071890</td>
          <td>0.047096</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.037448</td>
          <td>0.260712</td>
          <td>26.669438</td>
          <td>0.160167</td>
          <td>26.168705</td>
          <td>0.091485</td>
          <td>25.853589</td>
          <td>0.112726</td>
          <td>26.110254</td>
          <td>0.260966</td>
          <td>25.269071</td>
          <td>0.278107</td>
          <td>0.002237</td>
          <td>0.001409</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.594142</td>
          <td>0.405663</td>
          <td>26.435267</td>
          <td>0.131061</td>
          <td>26.743015</td>
          <td>0.150859</td>
          <td>26.332952</td>
          <td>0.170548</td>
          <td>25.924505</td>
          <td>0.224071</td>
          <td>25.840488</td>
          <td>0.436172</td>
          <td>0.008095</td>
          <td>0.007704</td>
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
