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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.15/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f481ae520b0>



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
    0      0.890625  27.370831  26.712660  26.025223  25.327185  25.016500   
    1      1.978239  29.557047  28.361183  27.587227  27.238544  26.628105   
    2      0.974287  26.566013  25.937716  24.787411  23.872454  23.139563   
    3      1.317978  29.042736  28.274597  27.501110  26.648792  26.091452   
    4      1.386366  26.292624  25.774778  25.429960  24.806530  24.367950   
    ...         ...        ...        ...        ...        ...        ...   
    99995  2.147172  26.550978  26.349937  26.135286  26.082020  25.911032   
    99996  1.457508  27.362209  27.036276  26.823141  26.420132  26.110037   
    99997  1.372993  27.736042  27.271955  26.887583  26.416138  26.043432   
    99998  0.855022  28.044554  27.327116  26.599014  25.862329  25.592169   
    99999  1.723768  27.049067  26.526747  26.094597  25.642973  25.197958   
    
                   y     major     minor  
    0      24.926819  0.003319  0.002869  
    1      26.248560  0.008733  0.007945  
    2      22.832047  0.103938  0.052162  
    3      25.346504  0.147522  0.143359  
    4      23.700008  0.010929  0.009473  
    ...          ...       ...       ...  
    99995  25.558136  0.086491  0.071701  
    99996  25.524906  0.044537  0.022302  
    99997  25.456163  0.073146  0.047825  
    99998  25.506388  0.100551  0.094662  
    99999  24.900501  0.059611  0.049181  
    
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
          <td>0.890625</td>
          <td>27.150209</td>
          <td>0.610744</td>
          <td>26.505101</td>
          <td>0.139101</td>
          <td>26.086785</td>
          <td>0.085118</td>
          <td>25.321904</td>
          <td>0.070626</td>
          <td>24.921112</td>
          <td>0.094666</td>
          <td>24.828564</td>
          <td>0.193114</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.723975</td>
          <td>0.895025</td>
          <td>27.772877</td>
          <td>0.394752</td>
          <td>27.466115</td>
          <td>0.276140</td>
          <td>26.702564</td>
          <td>0.232440</td>
          <td>27.194716</td>
          <td>0.599649</td>
          <td>26.100133</td>
          <td>0.528690</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.263140</td>
          <td>0.660732</td>
          <td>25.847585</td>
          <td>0.078355</td>
          <td>24.816186</td>
          <td>0.027666</td>
          <td>23.885901</td>
          <td>0.019991</td>
          <td>23.153530</td>
          <td>0.020011</td>
          <td>22.842649</td>
          <td>0.033936</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.291639</td>
          <td>0.673807</td>
          <td>29.607474</td>
          <td>1.320410</td>
          <td>27.127710</td>
          <td>0.208868</td>
          <td>26.804497</td>
          <td>0.252821</td>
          <td>26.058940</td>
          <td>0.250204</td>
          <td>25.243548</td>
          <td>0.272385</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.902318</td>
          <td>0.511038</td>
          <td>25.812250</td>
          <td>0.075951</td>
          <td>25.473178</td>
          <td>0.049431</td>
          <td>24.774921</td>
          <td>0.043471</td>
          <td>24.366573</td>
          <td>0.057997</td>
          <td>23.633893</td>
          <td>0.068429</td>
          <td>0.010929</td>
          <td>0.009473</td>
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
          <td>2.147172</td>
          <td>26.583726</td>
          <td>0.402225</td>
          <td>26.361560</td>
          <td>0.122868</td>
          <td>25.996119</td>
          <td>0.078577</td>
          <td>26.183399</td>
          <td>0.149955</td>
          <td>25.664137</td>
          <td>0.179933</td>
          <td>26.086725</td>
          <td>0.523543</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.657686</td>
          <td>0.425629</td>
          <td>26.751510</td>
          <td>0.171759</td>
          <td>26.826211</td>
          <td>0.161861</td>
          <td>26.449446</td>
          <td>0.188086</td>
          <td>26.126803</td>
          <td>0.264508</td>
          <td>27.730694</td>
          <td>1.453968</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.196321</td>
          <td>0.249179</td>
          <td>26.939367</td>
          <td>0.178222</td>
          <td>26.369615</td>
          <td>0.175794</td>
          <td>26.526448</td>
          <td>0.364151</td>
          <td>25.189276</td>
          <td>0.260586</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.028314</td>
          <td>1.076168</td>
          <td>28.090316</td>
          <td>0.501618</td>
          <td>26.558104</td>
          <td>0.128520</td>
          <td>25.927800</td>
          <td>0.120242</td>
          <td>25.675062</td>
          <td>0.181605</td>
          <td>24.836803</td>
          <td>0.194459</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.020297</td>
          <td>0.257076</td>
          <td>26.564500</td>
          <td>0.146394</td>
          <td>26.103561</td>
          <td>0.086385</td>
          <td>25.728380</td>
          <td>0.101038</td>
          <td>25.407491</td>
          <td>0.144518</td>
          <td>25.045785</td>
          <td>0.231550</td>
          <td>0.059611</td>
          <td>0.049181</td>
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
          <td>0.890625</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.725672</td>
          <td>0.192846</td>
          <td>26.082717</td>
          <td>0.099702</td>
          <td>25.302175</td>
          <td>0.082234</td>
          <td>24.925755</td>
          <td>0.111640</td>
          <td>24.744381</td>
          <td>0.211043</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.826025</td>
          <td>0.424005</td>
          <td>27.118923</td>
          <td>0.379033</td>
          <td>26.820042</td>
          <td>0.522714</td>
          <td>27.435418</td>
          <td>1.377315</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.400242</td>
          <td>0.802447</td>
          <td>26.033283</td>
          <td>0.108561</td>
          <td>24.769703</td>
          <td>0.031947</td>
          <td>23.871371</td>
          <td>0.023831</td>
          <td>23.180713</td>
          <td>0.024540</td>
          <td>22.779285</td>
          <td>0.038876</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.514482</td>
          <td>0.386740</td>
          <td>27.489668</td>
          <td>0.346543</td>
          <td>27.153689</td>
          <td>0.413678</td>
          <td>25.803005</td>
          <td>0.251032</td>
          <td>25.034188</td>
          <td>0.285726</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>28.357152</td>
          <td>1.388942</td>
          <td>25.721323</td>
          <td>0.080967</td>
          <td>25.383123</td>
          <td>0.053767</td>
          <td>24.806861</td>
          <td>0.053061</td>
          <td>24.346033</td>
          <td>0.067069</td>
          <td>23.836518</td>
          <td>0.096788</td>
          <td>0.010929</td>
          <td>0.009473</td>
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
          <td>2.147172</td>
          <td>30.474057</td>
          <td>3.226571</td>
          <td>26.391333</td>
          <td>0.147816</td>
          <td>26.054620</td>
          <td>0.099338</td>
          <td>26.138688</td>
          <td>0.173639</td>
          <td>25.887533</td>
          <td>0.257720</td>
          <td>25.520018</td>
          <td>0.401923</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.951862</td>
          <td>0.584835</td>
          <td>27.092803</td>
          <td>0.262467</td>
          <td>26.659354</td>
          <td>0.164895</td>
          <td>26.260321</td>
          <td>0.189272</td>
          <td>25.889564</td>
          <td>0.254134</td>
          <td>25.416205</td>
          <td>0.365224</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.381181</td>
          <td>0.788240</td>
          <td>27.139152</td>
          <td>0.274540</td>
          <td>27.161895</td>
          <td>0.253266</td>
          <td>26.526320</td>
          <td>0.238362</td>
          <td>26.406552</td>
          <td>0.386996</td>
          <td>25.936343</td>
          <td>0.544671</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.066233</td>
          <td>1.205209</td>
          <td>27.841503</td>
          <td>0.481973</td>
          <td>26.570091</td>
          <td>0.156892</td>
          <td>25.839189</td>
          <td>0.135764</td>
          <td>26.051839</td>
          <td>0.297351</td>
          <td>25.165284</td>
          <td>0.307061</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>29.545660</td>
          <td>2.362438</td>
          <td>26.209225</td>
          <td>0.125112</td>
          <td>26.075362</td>
          <td>0.100059</td>
          <td>25.783791</td>
          <td>0.126625</td>
          <td>25.241997</td>
          <td>0.148250</td>
          <td>25.016010</td>
          <td>0.266702</td>
          <td>0.059611</td>
          <td>0.049181</td>
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
          <td>0.890625</td>
          <td>26.041725</td>
          <td>0.261638</td>
          <td>26.880343</td>
          <td>0.191564</td>
          <td>26.128594</td>
          <td>0.088322</td>
          <td>25.533794</td>
          <td>0.085173</td>
          <td>25.059206</td>
          <td>0.106851</td>
          <td>24.727243</td>
          <td>0.177285</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.394448</td>
          <td>0.722880</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.498880</td>
          <td>0.283825</td>
          <td>28.663875</td>
          <td>0.974621</td>
          <td>26.057582</td>
          <td>0.250146</td>
          <td>26.178202</td>
          <td>0.559894</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>25.939721</td>
          <td>0.254171</td>
          <td>26.055102</td>
          <td>0.101048</td>
          <td>24.799276</td>
          <td>0.029591</td>
          <td>23.861251</td>
          <td>0.021282</td>
          <td>23.141186</td>
          <td>0.021449</td>
          <td>22.814364</td>
          <td>0.036056</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.649502</td>
          <td>0.427743</td>
          <td>27.593169</td>
          <td>0.374610</td>
          <td>29.242587</td>
          <td>1.548274</td>
          <td>26.174681</td>
          <td>0.337683</td>
          <td>25.276577</td>
          <td>0.345615</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.578954</td>
          <td>0.178189</td>
          <td>25.829699</td>
          <td>0.077225</td>
          <td>25.457916</td>
          <td>0.048836</td>
          <td>24.879408</td>
          <td>0.047767</td>
          <td>24.316763</td>
          <td>0.055569</td>
          <td>23.757448</td>
          <td>0.076445</td>
          <td>0.010929</td>
          <td>0.009473</td>
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
          <td>2.147172</td>
          <td>26.862036</td>
          <td>0.519821</td>
          <td>26.488754</td>
          <td>0.146748</td>
          <td>26.149948</td>
          <td>0.097358</td>
          <td>26.009342</td>
          <td>0.139946</td>
          <td>26.397183</td>
          <td>0.352956</td>
          <td>25.713731</td>
          <td>0.424727</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.662686</td>
          <td>0.868220</td>
          <td>26.791199</td>
          <td>0.180114</td>
          <td>27.090326</td>
          <td>0.205634</td>
          <td>26.489648</td>
          <td>0.197818</td>
          <td>26.267051</td>
          <td>0.300871</td>
          <td>25.499524</td>
          <td>0.339752</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.887040</td>
          <td>0.520067</td>
          <td>29.737913</td>
          <td>1.449347</td>
          <td>26.834999</td>
          <td>0.171002</td>
          <td>26.395592</td>
          <td>0.188794</td>
          <td>26.174577</td>
          <td>0.287681</td>
          <td>25.027538</td>
          <td>0.239235</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.981058</td>
          <td>1.105424</td>
          <td>27.250589</td>
          <td>0.286003</td>
          <td>26.845846</td>
          <td>0.183889</td>
          <td>25.842537</td>
          <td>0.125698</td>
          <td>25.224772</td>
          <td>0.138150</td>
          <td>25.445890</td>
          <td>0.356718</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.473085</td>
          <td>0.777149</td>
          <td>26.348167</td>
          <td>0.125587</td>
          <td>26.110926</td>
          <td>0.090406</td>
          <td>25.524666</td>
          <td>0.088008</td>
          <td>25.376686</td>
          <td>0.146198</td>
          <td>24.929186</td>
          <td>0.218350</td>
          <td>0.059611</td>
          <td>0.049181</td>
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




.. image:: ../../../docs/rendered/creation_examples/photometric_realization_demo_files/../../../docs/rendered/creation_examples/photometric_realization_demo_24_0.png


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




.. image:: ../../../docs/rendered/creation_examples/photometric_realization_demo_files/../../../docs/rendered/creation_examples/photometric_realization_demo_25_0.png


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
