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

    <pzflow.flow.Flow at 0x7f7011e605b0>



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
          <td>28.671714</td>
          <td>1.522422</td>
          <td>27.079281</td>
          <td>0.226219</td>
          <td>26.034693</td>
          <td>0.081298</td>
          <td>25.359917</td>
          <td>0.073042</td>
          <td>25.069702</td>
          <td>0.107821</td>
          <td>24.523400</td>
          <td>0.148953</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>26.654365</td>
          <td>0.424555</td>
          <td>28.076708</td>
          <td>0.496606</td>
          <td>27.541593</td>
          <td>0.293537</td>
          <td>27.190594</td>
          <td>0.345017</td>
          <td>28.309874</td>
          <td>1.213837</td>
          <td>26.139763</td>
          <td>0.544134</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.209906</td>
          <td>0.636808</td>
          <td>25.848822</td>
          <td>0.078440</td>
          <td>24.777739</td>
          <td>0.026753</td>
          <td>23.893064</td>
          <td>0.020112</td>
          <td>23.157101</td>
          <td>0.020071</td>
          <td>22.739666</td>
          <td>0.030994</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.989241</td>
          <td>0.465331</td>
          <td>27.374625</td>
          <td>0.256273</td>
          <td>26.662134</td>
          <td>0.224773</td>
          <td>25.941959</td>
          <td>0.227161</td>
          <td>25.486735</td>
          <td>0.331186</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.361129</td>
          <td>0.338161</td>
          <td>25.910047</td>
          <td>0.082788</td>
          <td>25.412823</td>
          <td>0.046852</td>
          <td>24.801137</td>
          <td>0.044494</td>
          <td>24.313878</td>
          <td>0.055347</td>
          <td>23.765222</td>
          <td>0.076857</td>
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
          <td>27.030655</td>
          <td>0.560959</td>
          <td>26.334611</td>
          <td>0.120028</td>
          <td>26.178081</td>
          <td>0.092238</td>
          <td>26.076232</td>
          <td>0.136741</td>
          <td>25.880072</td>
          <td>0.215759</td>
          <td>25.354731</td>
          <td>0.298031</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.035410</td>
          <td>0.562878</td>
          <td>27.086793</td>
          <td>0.227634</td>
          <td>26.922252</td>
          <td>0.175653</td>
          <td>26.383039</td>
          <td>0.177808</td>
          <td>25.790792</td>
          <td>0.200223</td>
          <td>25.744463</td>
          <td>0.405037</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>28.407908</td>
          <td>1.329758</td>
          <td>27.109500</td>
          <td>0.231958</td>
          <td>26.948210</td>
          <td>0.179563</td>
          <td>26.241834</td>
          <td>0.157655</td>
          <td>26.118989</td>
          <td>0.262825</td>
          <td>25.726654</td>
          <td>0.399526</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.338244</td>
          <td>1.281044</td>
          <td>27.091151</td>
          <td>0.228458</td>
          <td>26.550770</td>
          <td>0.127706</td>
          <td>26.046419</td>
          <td>0.133264</td>
          <td>25.500026</td>
          <td>0.156462</td>
          <td>25.315433</td>
          <td>0.288735</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.046411</td>
          <td>0.567338</td>
          <td>27.150708</td>
          <td>0.239994</td>
          <td>26.063941</td>
          <td>0.083422</td>
          <td>25.453213</td>
          <td>0.079319</td>
          <td>25.213197</td>
          <td>0.122175</td>
          <td>24.813754</td>
          <td>0.190719</td>
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
          <td>26.541105</td>
          <td>0.431151</td>
          <td>26.585135</td>
          <td>0.171232</td>
          <td>25.913169</td>
          <td>0.085908</td>
          <td>25.337141</td>
          <td>0.084807</td>
          <td>25.126829</td>
          <td>0.132933</td>
          <td>24.764304</td>
          <td>0.214584</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.832153</td>
          <td>0.467308</td>
          <td>28.783883</td>
          <td>0.833121</td>
          <td>27.633015</td>
          <td>0.557214</td>
          <td>26.395519</td>
          <td>0.379468</td>
          <td>25.626576</td>
          <td>0.428031</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.316033</td>
          <td>0.759299</td>
          <td>25.796280</td>
          <td>0.088225</td>
          <td>24.704661</td>
          <td>0.030174</td>
          <td>23.899848</td>
          <td>0.024424</td>
          <td>23.093664</td>
          <td>0.022767</td>
          <td>22.827029</td>
          <td>0.040553</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>28.548067</td>
          <td>1.574550</td>
          <td>29.566908</td>
          <td>1.458308</td>
          <td>26.905500</td>
          <td>0.215608</td>
          <td>26.545451</td>
          <td>0.255209</td>
          <td>25.749182</td>
          <td>0.240151</td>
          <td>26.653496</td>
          <td>0.921479</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.140113</td>
          <td>0.315569</td>
          <td>25.636937</td>
          <td>0.075165</td>
          <td>25.439776</td>
          <td>0.056539</td>
          <td>24.749345</td>
          <td>0.050421</td>
          <td>24.335883</td>
          <td>0.066469</td>
          <td>23.875235</td>
          <td>0.100128</td>
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
          <td>26.307025</td>
          <td>0.365137</td>
          <td>26.280298</td>
          <td>0.134346</td>
          <td>26.088335</td>
          <td>0.102314</td>
          <td>26.350610</td>
          <td>0.207621</td>
          <td>25.806519</td>
          <td>0.241118</td>
          <td>25.010831</td>
          <td>0.268366</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.858869</td>
          <td>0.547111</td>
          <td>26.853713</td>
          <td>0.215456</td>
          <td>26.548427</td>
          <td>0.149968</td>
          <td>26.546256</td>
          <td>0.240297</td>
          <td>26.192527</td>
          <td>0.324638</td>
          <td>24.782516</td>
          <td>0.218754</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.186285</td>
          <td>0.692085</td>
          <td>27.613218</td>
          <td>0.399660</td>
          <td>26.947712</td>
          <td>0.212102</td>
          <td>26.684319</td>
          <td>0.271338</td>
          <td>25.670294</td>
          <td>0.213681</td>
          <td>25.961499</td>
          <td>0.554660</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.288448</td>
          <td>0.663927</td>
          <td>26.415166</td>
          <td>0.137339</td>
          <td>25.915066</td>
          <td>0.144936</td>
          <td>25.487321</td>
          <td>0.186534</td>
          <td>25.029355</td>
          <td>0.275154</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.321523</td>
          <td>0.756678</td>
          <td>26.452247</td>
          <td>0.154237</td>
          <td>26.238585</td>
          <td>0.115388</td>
          <td>25.729250</td>
          <td>0.120773</td>
          <td>25.422487</td>
          <td>0.172965</td>
          <td>25.112318</td>
          <td>0.288393</td>
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
          <td>30.827502</td>
          <td>3.419062</td>
          <td>26.794817</td>
          <td>0.178208</td>
          <td>25.809102</td>
          <td>0.066605</td>
          <td>25.296607</td>
          <td>0.069072</td>
          <td>25.060515</td>
          <td>0.106973</td>
          <td>25.115649</td>
          <td>0.245336</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.619659</td>
          <td>1.483972</td>
          <td>27.862305</td>
          <td>0.423078</td>
          <td>27.902049</td>
          <td>0.390612</td>
          <td>27.289482</td>
          <td>0.373153</td>
          <td>27.055393</td>
          <td>0.543133</td>
          <td>26.162033</td>
          <td>0.553410</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.875692</td>
          <td>1.022864</td>
          <td>25.853400</td>
          <td>0.084662</td>
          <td>24.794307</td>
          <td>0.029462</td>
          <td>23.885825</td>
          <td>0.021733</td>
          <td>23.151333</td>
          <td>0.021636</td>
          <td>22.831446</td>
          <td>0.036604</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.532331</td>
          <td>0.390993</td>
          <td>28.305359</td>
          <td>0.634384</td>
          <td>26.470325</td>
          <td>0.239087</td>
          <td>25.948830</td>
          <td>0.281812</td>
          <td>25.777354</td>
          <td>0.506426</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.638380</td>
          <td>0.419773</td>
          <td>25.717606</td>
          <td>0.069949</td>
          <td>25.380258</td>
          <td>0.045582</td>
          <td>24.867049</td>
          <td>0.047246</td>
          <td>24.336281</td>
          <td>0.056540</td>
          <td>23.758467</td>
          <td>0.076514</td>
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
          <td>26.827710</td>
          <td>0.506904</td>
          <td>26.175663</td>
          <td>0.111928</td>
          <td>26.237314</td>
          <td>0.105098</td>
          <td>25.868581</td>
          <td>0.123905</td>
          <td>25.927010</td>
          <td>0.241607</td>
          <td>25.866022</td>
          <td>0.476379</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.203924</td>
          <td>0.254116</td>
          <td>26.683087</td>
          <td>0.145496</td>
          <td>26.847105</td>
          <td>0.266040</td>
          <td>26.037568</td>
          <td>0.249660</td>
          <td>25.717019</td>
          <td>0.402559</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.485967</td>
          <td>0.384462</td>
          <td>28.179360</td>
          <td>0.554558</td>
          <td>27.070971</td>
          <td>0.208679</td>
          <td>26.510969</td>
          <td>0.208021</td>
          <td>26.296592</td>
          <td>0.317305</td>
          <td>25.652358</td>
          <td>0.394448</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.199638</td>
          <td>0.274431</td>
          <td>26.558903</td>
          <td>0.143951</td>
          <td>25.769308</td>
          <td>0.117953</td>
          <td>25.582727</td>
          <td>0.187538</td>
          <td>25.779720</td>
          <td>0.460925</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.288752</td>
          <td>0.686892</td>
          <td>26.722747</td>
          <td>0.173202</td>
          <td>26.041178</td>
          <td>0.085023</td>
          <td>25.713860</td>
          <td>0.103902</td>
          <td>25.388215</td>
          <td>0.147654</td>
          <td>24.682628</td>
          <td>0.177465</td>
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
