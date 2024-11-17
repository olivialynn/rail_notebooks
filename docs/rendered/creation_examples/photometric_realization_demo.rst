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

    <pzflow.flow.Flow at 0x7f951d2946d0>



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
          <td>27.037575</td>
          <td>0.563754</td>
          <td>26.962181</td>
          <td>0.205175</td>
          <td>26.113918</td>
          <td>0.087177</td>
          <td>25.384401</td>
          <td>0.074641</td>
          <td>25.074073</td>
          <td>0.108233</td>
          <td>24.770121</td>
          <td>0.183818</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.215034</td>
          <td>0.549419</td>
          <td>27.571971</td>
          <td>0.300804</td>
          <td>27.241758</td>
          <td>0.359177</td>
          <td>26.258087</td>
          <td>0.294241</td>
          <td>25.322578</td>
          <td>0.290406</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.863380</td>
          <td>0.975448</td>
          <td>25.894602</td>
          <td>0.081669</td>
          <td>24.809176</td>
          <td>0.027497</td>
          <td>23.863954</td>
          <td>0.019623</td>
          <td>23.151686</td>
          <td>0.019979</td>
          <td>22.903607</td>
          <td>0.035812</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>26.995166</td>
          <td>0.546793</td>
          <td>27.406974</td>
          <td>0.295769</td>
          <td>27.492166</td>
          <td>0.282039</td>
          <td>26.338851</td>
          <td>0.171258</td>
          <td>26.478066</td>
          <td>0.350591</td>
          <td>26.011000</td>
          <td>0.495204</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.353930</td>
          <td>0.336243</td>
          <td>25.803764</td>
          <td>0.075384</td>
          <td>25.457679</td>
          <td>0.048755</td>
          <td>24.909089</td>
          <td>0.048969</td>
          <td>24.436759</td>
          <td>0.061722</td>
          <td>23.828793</td>
          <td>0.081293</td>
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
          <td>26.600614</td>
          <td>0.407473</td>
          <td>26.453096</td>
          <td>0.132999</td>
          <td>26.222222</td>
          <td>0.095884</td>
          <td>26.390489</td>
          <td>0.178935</td>
          <td>25.775027</td>
          <td>0.197588</td>
          <td>25.423873</td>
          <td>0.315021</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>28.449010</td>
          <td>1.358936</td>
          <td>27.053311</td>
          <td>0.221391</td>
          <td>26.919863</td>
          <td>0.175297</td>
          <td>26.149844</td>
          <td>0.145694</td>
          <td>26.306907</td>
          <td>0.306020</td>
          <td>25.732139</td>
          <td>0.401217</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.174321</td>
          <td>0.621174</td>
          <td>27.589338</td>
          <td>0.342064</td>
          <td>27.143046</td>
          <td>0.211564</td>
          <td>26.349112</td>
          <td>0.172759</td>
          <td>25.840205</td>
          <td>0.208691</td>
          <td>26.512274</td>
          <td>0.706477</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.322895</td>
          <td>0.688360</td>
          <td>27.083829</td>
          <td>0.227075</td>
          <td>26.345782</td>
          <td>0.106841</td>
          <td>26.038962</td>
          <td>0.132408</td>
          <td>25.504370</td>
          <td>0.157045</td>
          <td>25.235996</td>
          <td>0.270715</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.020355</td>
          <td>0.556819</td>
          <td>26.719669</td>
          <td>0.167170</td>
          <td>26.228584</td>
          <td>0.096420</td>
          <td>25.567536</td>
          <td>0.087729</td>
          <td>25.041655</td>
          <td>0.105211</td>
          <td>25.067870</td>
          <td>0.235822</td>
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
          <td>26.509961</td>
          <td>0.421058</td>
          <td>26.925208</td>
          <td>0.227840</td>
          <td>25.939567</td>
          <td>0.087927</td>
          <td>25.423518</td>
          <td>0.091502</td>
          <td>24.913155</td>
          <td>0.110420</td>
          <td>24.990813</td>
          <td>0.258773</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.176005</td>
          <td>0.682161</td>
          <td>28.435639</td>
          <td>0.718077</td>
          <td>27.067619</td>
          <td>0.231576</td>
          <td>26.908861</td>
          <td>0.321280</td>
          <td>26.350376</td>
          <td>0.366356</td>
          <td>26.593401</td>
          <td>0.844616</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>25.809567</td>
          <td>0.245221</td>
          <td>25.945838</td>
          <td>0.100581</td>
          <td>24.774787</td>
          <td>0.032090</td>
          <td>23.896700</td>
          <td>0.024358</td>
          <td>23.162347</td>
          <td>0.024153</td>
          <td>22.816840</td>
          <td>0.040189</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>28.568322</td>
          <td>1.590085</td>
          <td>27.944570</td>
          <td>0.534239</td>
          <td>26.833235</td>
          <td>0.202962</td>
          <td>26.641653</td>
          <td>0.276054</td>
          <td>26.086442</td>
          <td>0.315832</td>
          <td>25.759160</td>
          <td>0.501219</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.283565</td>
          <td>0.353490</td>
          <td>25.723950</td>
          <td>0.081155</td>
          <td>25.308982</td>
          <td>0.050344</td>
          <td>24.809087</td>
          <td>0.053166</td>
          <td>24.342150</td>
          <td>0.066838</td>
          <td>23.840009</td>
          <td>0.097085</td>
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
          <td>26.928601</td>
          <td>0.581250</td>
          <td>26.498483</td>
          <td>0.162007</td>
          <td>26.023535</td>
          <td>0.096668</td>
          <td>25.981749</td>
          <td>0.151870</td>
          <td>25.898886</td>
          <td>0.260126</td>
          <td>26.458616</td>
          <td>0.786288</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>28.297024</td>
          <td>1.348257</td>
          <td>27.375378</td>
          <td>0.329552</td>
          <td>26.780146</td>
          <td>0.182713</td>
          <td>26.556673</td>
          <td>0.242371</td>
          <td>26.379495</td>
          <td>0.376086</td>
          <td>28.270703</td>
          <td>2.038832</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.329457</td>
          <td>0.319980</td>
          <td>27.129443</td>
          <td>0.246601</td>
          <td>26.051559</td>
          <td>0.159893</td>
          <td>26.079802</td>
          <td>0.298978</td>
          <td>25.213975</td>
          <td>0.313752</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.471835</td>
          <td>0.844694</td>
          <td>27.684167</td>
          <td>0.428196</td>
          <td>26.725766</td>
          <td>0.179133</td>
          <td>25.753962</td>
          <td>0.126115</td>
          <td>25.240618</td>
          <td>0.151198</td>
          <td>26.615098</td>
          <td>0.876428</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.990012</td>
          <td>0.603031</td>
          <td>26.287275</td>
          <td>0.133846</td>
          <td>26.078673</td>
          <td>0.100350</td>
          <td>25.582119</td>
          <td>0.106242</td>
          <td>25.138057</td>
          <td>0.135559</td>
          <td>25.067110</td>
          <td>0.278025</td>
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
          <td>28.058440</td>
          <td>1.095268</td>
          <td>26.770448</td>
          <td>0.174563</td>
          <td>26.034529</td>
          <td>0.081297</td>
          <td>25.371980</td>
          <td>0.073836</td>
          <td>25.088364</td>
          <td>0.109607</td>
          <td>25.274307</td>
          <td>0.279313</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>26.908747</td>
          <td>0.513734</td>
          <td>30.608954</td>
          <td>2.109075</td>
          <td>27.272161</td>
          <td>0.235750</td>
          <td>27.322489</td>
          <td>0.382852</td>
          <td>26.524776</td>
          <td>0.363981</td>
          <td>25.363700</td>
          <td>0.300460</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.197377</td>
          <td>0.313064</td>
          <td>25.942754</td>
          <td>0.091576</td>
          <td>24.796980</td>
          <td>0.029531</td>
          <td>23.880977</td>
          <td>0.021644</td>
          <td>23.137155</td>
          <td>0.021375</td>
          <td>22.854374</td>
          <td>0.037353</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.611791</td>
          <td>0.380073</td>
          <td>26.594941</td>
          <td>0.264845</td>
          <td>25.822302</td>
          <td>0.254190</td>
          <td>25.079439</td>
          <td>0.295362</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.734048</td>
          <td>0.451331</td>
          <td>25.710209</td>
          <td>0.069493</td>
          <td>25.426048</td>
          <td>0.047473</td>
          <td>24.845301</td>
          <td>0.046343</td>
          <td>24.375561</td>
          <td>0.058545</td>
          <td>23.646865</td>
          <td>0.069323</td>
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
          <td>26.352929</td>
          <td>0.353235</td>
          <td>26.157807</td>
          <td>0.110200</td>
          <td>26.140354</td>
          <td>0.096542</td>
          <td>26.096859</td>
          <td>0.150885</td>
          <td>25.764214</td>
          <td>0.211057</td>
          <td>27.463473</td>
          <td>1.328001</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.742427</td>
          <td>0.458277</td>
          <td>27.099310</td>
          <td>0.233134</td>
          <td>26.565630</td>
          <td>0.131477</td>
          <td>26.334980</td>
          <td>0.173574</td>
          <td>25.604530</td>
          <td>0.173799</td>
          <td>27.809176</td>
          <td>1.527073</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.265226</td>
          <td>0.679723</td>
          <td>27.252059</td>
          <td>0.271377</td>
          <td>26.892819</td>
          <td>0.179605</td>
          <td>26.182076</td>
          <td>0.157461</td>
          <td>25.792476</td>
          <td>0.210069</td>
          <td>25.335515</td>
          <td>0.307390</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.256184</td>
          <td>0.287300</td>
          <td>26.457038</td>
          <td>0.131841</td>
          <td>25.720755</td>
          <td>0.113071</td>
          <td>25.904186</td>
          <td>0.245224</td>
          <td>25.059899</td>
          <td>0.261790</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.596540</td>
          <td>0.415969</td>
          <td>26.500227</td>
          <td>0.143203</td>
          <td>26.136402</td>
          <td>0.092453</td>
          <td>25.487224</td>
          <td>0.085154</td>
          <td>25.177669</td>
          <td>0.123104</td>
          <td>24.765903</td>
          <td>0.190415</td>
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
